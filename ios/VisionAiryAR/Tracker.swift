import ARKit
import SceneKit
import SwiftUI
import simd

/// ARKit tracks the phone (camera + motion sensors, millimetre-level), the beacon streams its own
/// IMU-tracked position, and calibration ties the two together.
///
/// Graph frame (shared with the beacon and /graph on the laptop): origin = calibration spot, Z up,
/// Y = the beacon's Y arrow (= the phone's top edge at calibration), X = right. `graphRoot` is a node
/// placed in the AR world with exactly that frame, so beacon coordinates go straight onto its children.
final class Tracker: NSObject, ObservableObject, ARSCNViewDelegate, ARSessionDelegate {
    // What the overlay shows
    @Published var linkText = "CONNECTING"
    @Published var phase = "UNCAL"
    @Published var beaconText = "--"
    @Published var camText = "--"
    @Published var distText = "--"
    @Published var throwText = ""
    @Published var toast = ""
    @Published var countdown: Int?
    @Published var trackingText = ""

    let sceneView = ARSCNView()
    private let link: BeaconLink

    private let graphRoot = SCNNode()
    private let beaconNode = SCNNode()
    private let trailRoot = SCNNode()
    private var dropNode: SCNNode?
    private var trail: [(t: TimeInterval, node: SCNNode)] = []
    private var lastTrailPoint: simd_float3?
    private var calibrated = false

    // Latest beacon data
    private var beaconPos: simd_float3?
    private var beaconStill = false
    private var beaconAir = false
    private var beaconCal = false
    private var heartbeat: [String: Any] = [:]
    private var heartbeatAt: TimeInterval = 0

    // Auto-calibration: phone held flat, screen up, with the beacon lying on it
    private var flatSince: TimeInterval?
    private var flatPos = simd_float3()
    private var flatTop = simd_float3()
    private var flatDone = false
    private var pendingCal: (pos: simd_float3, top: simd_float3)?

    private var lastThrow: (v: [Double], p0: [Double], peak: Double, air: Double?, hit: Double?, p1: simd_float3?)?
    private var lastPoseSent: TimeInterval = 0
    private var toastUntil: TimeInterval = 0

    static let trailSeconds: TimeInterval = 6
    static let autoCalSeconds: TimeInterval = 2
    private static let green = UIColor(red: 0.22, green: 1, blue: 0.42, alpha: 1)
    private static let yellow = UIColor(red: 1, green: 0.82, blue: 0.25, alpha: 1)

    override init() {
        let host = Bundle.main.object(forInfoDictionaryKey: "VAServerHost") as? String ?? "127.0.0.1"
        link = BeaconLink(host: host)
        super.init()
        buildScene()
        link.onMessage = { [weak self] msg in self?.handle(msg) }
    }

    func start() {
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.automaticallyUpdatesLighting = true
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]
        sceneView.session.run(config)
        link.connect()
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.tick() }
    }

    // MARK: - Scene

    private func buildScene() {
        sceneView.scene.rootNode.addChildNode(graphRoot)
        graphRoot.isHidden = true
        graphRoot.addChildNode(trailRoot)

        // Origin axes: X red, Y green (the beacon's arrow), Z blue (up)
        for (axis, color) in [(simd_float3(1, 0, 0), UIColor.systemRed), (simd_float3(0, 1, 0), Self.green), (simd_float3(0, 0, 1), UIColor.systemBlue)] {
            graphRoot.addChildNode(Self.segment(from: .zero, to: axis * 0.25, radius: 0.004, color: color))
        }
        // Grid on the calibration plane (Z = 0), 2 m around the origin
        let gridColor = Self.green.withAlphaComponent(0.25)
        for i in stride(from: Float(-1), through: 1, by: 0.25) {
            graphRoot.addChildNode(Self.segment(from: [i, -1, 0], to: [i, 1, 0], radius: 0.0012, color: gridColor))
            graphRoot.addChildNode(Self.segment(from: [-1, i, 0], to: [1, i, 0], radius: 0.0012, color: gridColor))
        }

        // The beacon: a small board with its Y arrow, plus a glow sphere so it's easy to see
        let board = SCNNode(geometry: SCNBox(width: 0.055, height: 0.08, length: 0.012, chamferRadius: 0.002))
        board.geometry?.firstMaterial = Self.material(Self.yellow)
        board.addChildNode(Self.segment(from: [0, -0.03, 0.007], to: [0, 0.04, 0.007], radius: 0.003, color: .black))
        beaconNode.addChildNode(board)
        let glow = SCNNode(geometry: SCNSphere(radius: 0.05))
        glow.geometry?.firstMaterial = Self.material(Self.yellow.withAlphaComponent(0.18))
        beaconNode.addChildNode(glow)
        beaconNode.isHidden = true
        graphRoot.addChildNode(beaconNode)
    }

    private static func material(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.diffuse.contents = color
        m.emission.contents = color
        m.lightingModel = .constant
        m.isDoubleSided = true
        return m
    }

    private static var materials: [UIColor: SCNMaterial] = [:]

    /// A thin cylinder from a to b (in the parent's coordinates)
    static func segment(from a: simd_float3, to b: simd_float3, radius: CGFloat, color: UIColor) -> SCNNode {
        let d = b - a
        let length = simd_length(d)
        let cylinder = SCNCylinder(radius: radius, height: CGFloat(max(length, 0.0005)))
        cylinder.radialSegmentCount = 6
        if materials[color] == nil { materials[color] = material(color) }
        cylinder.firstMaterial = materials[color]
        let node = SCNNode(geometry: cylinder)
        node.simdPosition = (a + b) / 2
        if length > 0 { node.simdOrientation = simd_quatf(from: [0, 1, 0], to: d / length) }
        return node
    }

    // ARKit's detected surfaces, drawn faintly so you can see what it found
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let plane = anchor as? ARPlaneAnchor, let device = sceneView.device,
              let geometry = ARSCNPlaneGeometry(device: device) else { return nil }
        geometry.update(from: plane.geometry)
        geometry.firstMaterial = Self.material(UIColor(red: 0.2, green: 0.7, blue: 1, alpha: 0.12))
        return SCNNode(geometry: geometry)
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let plane = anchor as? ARPlaneAnchor, let geometry = node.geometry as? ARSCNPlaneGeometry else { return }
        geometry.update(from: plane.geometry)
    }

    // MARK: - Messages from the server

    private func handle(_ msg: [String: Any]) {
        switch msg["type"] as? String {
        case "bstate": onBeaconState(msg)
        case "beacon":
            if msg["online"] as? Bool == false { heartbeat = [:]; heartbeatAt = 0 } else { heartbeat = msg; heartbeatAt = now }
        case "imu": onEvent(msg)
        default: break
        }
    }

    private var now: TimeInterval { CACurrentMediaTime() }

    private func onBeaconState(_ msg: [String: Any]) {
        guard let p = msg["p"] as? [Double], p.count == 3, let q = msg["q"] as? [Double], q.count == 4 else { return }
        let pos = simd_float3(Float(p[0]), Float(p[1]), Float(p[2]))
        beaconPos = pos
        beaconStill = msg["still"] as? Bool ?? false
        beaconAir = msg["air"] as? Bool ?? false
        beaconCal = msg["cal"] as? Bool ?? false
        guard calibrated, beaconCal else { return }

        beaconNode.isHidden = false
        beaconNode.simdPosition = pos
        beaconNode.simdOrientation = simd_quatf(ix: Float(q[1]), iy: Float(q[2]), iz: Float(q[3]), r: Float(q[0]))

        // Trail: add a segment once it has moved at least 5 mm, yellow while flying
        if let last = lastTrailPoint {
            if simd_distance(last, pos) > 0.005 {
                let seg = Self.segment(from: last, to: pos, radius: 0.004, color: beaconAir ? Self.yellow : Self.green)
                trailRoot.addChildNode(seg)
                trail.append((now, seg))
                lastTrailPoint = pos
            }
        } else {
            lastTrailPoint = pos
        }
        while let first = trail.first, now - first.t > Self.trailSeconds {
            first.node.removeFromParentNode()
            trail.removeFirst()
        }

        // Dashed-looking drop line to the Z = 0 plane, to show height
        dropNode?.removeFromParentNode()
        dropNode = Self.segment(from: pos, to: [pos.x, pos.y, 0], radius: 0.0015, color: UIColor.white.withAlphaComponent(0.35))
        graphRoot.addChildNode(dropNode!)

        if beaconAir, var t = lastThrow, t.air == nil {
            t.peak = max(t.peak, Double(pos.z))
            lastThrow = t
        }
    }

    private func onEvent(_ msg: [String: Any]) {
        switch msg["event"] as? String {
        case "cal":
            if msg["result"] as? String == "ok", let cal = pendingCal {
                setGraphFrame(origin: cal.pos, top: cal.top)
                calibrated = true
                show("CALIBRATED // ORIGIN SET")
            } else if msg["result"] as? String == "moving" {
                show("BEACON MOVED // TRY AGAIN")
            }
            pendingCal = nil
        case "zero":
            clearTrail()
        case "launch":
            let v = msg["v"] as? [Double] ?? [0, 0, 0], p0 = msg["d"] as? [Double] ?? [0, 0, 0]
            lastThrow = (v, p0, p0[2], nil, nil, nil)
            show("LAUNCH")
        case "land":
            guard var t = lastThrow, t.air == nil else { return }
            t.air = (msg["airtime"] as? Double ?? 0) / 1000
            t.hit = msg["impactG"] as? Double
            t.p1 = beaconPos
            lastThrow = t
            show("IMPACT")
        default: break
        }
    }

    // MARK: - Calibration and re-zero

    /// The phone's pose with portrait axes: x right, y toward the top edge, z out of the screen
    private func portraitPose(_ frame: ARFrame) -> simd_float4x4 {
        frame.camera.viewMatrix(for: .portrait).inverse
    }

    private func setGraphFrame(origin: simd_float3, top: simd_float3) {
        // Y = the phone's top edge flattened onto the horizontal, Z = up, X = Y × Z
        var fwd = simd_float3(top.x, 0, top.z)
        fwd = simd_length(fwd) > 0.01 ? simd_normalize(fwd) : simd_float3(0, 0, -1)
        let up = simd_float3(0, 1, 0)
        let right = simd_cross(fwd, up)
        graphRoot.simdTransform = simd_float4x4(columns: (
            simd_float4(right, 0), simd_float4(fwd, 0), simd_float4(up, 0), simd_float4(origin, 1)))
        graphRoot.isHidden = false
        clearTrail()
    }

    func calibrateNow() {
        guard let frame = sceneView.session.currentFrame else { return }
        let pose = portraitPose(frame)
        guard pose.columns.2.y > 0.85 else { return show("HOLD THE PHONE FLAT, SCREEN UP") }
        guard beaconLive else { return show("NO BEACON LINK") }
        pendingCal = (simd_make_float3(pose.columns.3), simd_make_float3(pose.columns.1))
        link.send(["type": "cmd", "cmd": "cal"])
        show("CALIBRATING // HOLD STILL")
    }

    /// Beacon touching the phone: both are at the same point, so move the origin here and zero the beacon.
    /// The axes stay as calibrated. This wipes out the beacon's accumulated drift.
    func rezero() {
        guard calibrated, let frame = sceneView.session.currentFrame else { return show("CALIBRATE FIRST") }
        var m = graphRoot.simdTransform
        m.columns.3 = simd_float4(simd_make_float3(portraitPose(frame).columns.3), 1)
        graphRoot.simdTransform = m
        link.send(["type": "cmd", "cmd": "zero"])
        clearTrail()
        show("RE-ZEROED")
    }

    func clearTrail() {
        trail.forEach { $0.node.removeFromParentNode() }
        trail.removeAll()
        lastTrailPoint = nil
        lastThrow = nil
    }

    private var beaconLive: Bool { now - heartbeatAt < 3 }

    private func autoCalibrate(_ frame: ARFrame) {
        let pose = portraitPose(frame)
        let pos = simd_make_float3(pose.columns.3), top = simd_make_float3(pose.columns.1)
        let flat = pose.columns.2.y > 0.93
        let trackingOK: Bool = { if case .normal = frame.camera.trackingState { return true }; return false }()
        let beaconQuiet = beaconLive && (heartbeat["spin"] as? Double ?? 99) < 12 && !(heartbeat["airborne"] as? Bool ?? false)
        guard flat, trackingOK, beaconQuiet else {
            if !flat { flatDone = false }
            flatSince = nil
            countdown = nil
            return
        }
        // Held by hand, so allow a little wobble, but restart the countdown if it really moves
        if flatSince == nil || simd_distance(pos, flatPos) > 0.03 || simd_dot(simd_normalize(top), simd_normalize(flatTop)) < 0.998 {
            flatSince = now
            flatPos = pos
            flatTop = top
        }
        guard !flatDone, let since = flatSince else { countdown = nil; return }
        let left = Self.autoCalSeconds - (now - since)
        if left <= 0 {
            flatDone = true
            countdown = nil
            calibrateNow()
        } else {
            countdown = Int(ceil(left))
        }
    }

    // MARK: - Per frame

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        autoCalibrate(frame)
        guard calibrated else { return }
        // Camera pose in the graph frame
        let cam = graphRoot.simdWorldTransform.inverse * portraitPose(frame)
        let p = simd_make_float3(cam.columns.3)
        camText = Self.xyz(p)
        if let b = beaconPos { distText = String(format: "%.2f m", simd_distance(b, p)) }
        if now - lastPoseSent > 0.05 {
            lastPoseSent = now
            let col = { (i: Int) -> [Float] in let c = cam[i]; return [c.x, c.y, c.z] }
            link.send(["type": "phone", "p": [p.x, p.y, p.z], "x": col(0), "y": col(1), "z": col(2)])
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        switch camera.trackingState {
        case .normal: trackingText = ""
        case .notAvailable: trackingText = "AR TRACKING UNAVAILABLE"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: trackingText = "AR: SLOW DOWN"
            case .insufficientFeatures: trackingText = "AR: POINT AT SOMETHING WITH TEXTURE"
            case .initializing, .relocalizing: trackingText = "AR: MOVE THE PHONE AROUND A BIT"
            @unknown default: trackingText = "AR TRACKING LIMITED"
            }
        }
    }

    // MARK: - Overlay text

    private func tick() {
        let hb = beaconLive ? heartbeat : [:]
        linkText = !link.connected ? "NO SERVER" : hb.isEmpty ? "NO BEACON" : (hb["link"] as? String == "usb" ? "BEACON USB" : "BEACON WI-FI")
        phase = hb.isEmpty ? "--" : !calibrated || !beaconCal ? "UNCAL" : beaconAir ? "FLIGHT" : beaconStill ? "STILL" : "MOVING"
        beaconText = beaconPos.map(Self.xyz) ?? "--"
        if let t = lastThrow {
            let horiz = hypot(t.v[0], t.v[1])
            var parts = [String(format: "V0 %.1f m/s @ %.0f°", hypot(horiz, t.v[2]), atan2(t.v[2], horiz) * 180 / .pi),
                         String(format: "PEAK Z %+.2f", t.peak)]
            if let air = t.air, let p1 = t.p1 {
                parts.append(String(format: "AIR %.2fs", air))
                parts.append(String(format: "DIST %.2fm", hypot(Double(p1.x) - t.p0[0], Double(p1.y) - t.p0[1])))
                if let hit = t.hit { parts.append(String(format: "HIT %.1fg", hit)) }
            } else {
                parts.append("IN FLIGHT")
            }
            throwText = parts.joined(separator: "   ")
        } else {
            throwText = ""
        }
        if now > toastUntil { toast = "" }
    }

    private func show(_ text: String) {
        toast = text
        toastUntil = now + 2
    }

    static func xyz(_ p: simd_float3) -> String {
        String(format: "X%+.2f Y%+.2f Z%+.2f", p.x, p.y, p.z)
    }
}
