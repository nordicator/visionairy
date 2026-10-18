import ARKit
import SceneKit
import SwiftUI
import simd

/// ARKit tracks the phone (camera + motion sensors, millimetre-level), the beacon streams its own
/// IMU-tracked position, and calibration ties the two together.
///
/// Graph frame (shared with the beacon and /graph on the laptop): origin = where the beacon sat at sync,
/// Z up, Y = the beacon's Y arrow (= the phone's top edge at sync), X = right. `graphRoot` is a node
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
    @Published var pathText = ""

    let airpod = AirPodMotion()

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

    // Sync: phone flat with the beacon beside its right edge. The phone's pose is captured at the tap, then
    // the beacon has to stay put for syncSeconds while it calibrates; the phone is free to move meanwhile.
    private var pendingCal: (pos: simd_float3, top: simd_float3)?
    private var syncStarted: TimeInterval?
    private var calSent = false

    // Throw path: phone touched to the bag at release and at landing, AirPod timing the flight
    private var throwStart: (pos: simd_float3, at: TimeInterval)?
    private let pathRoot = SCNNode()

    private var lastThrow: (v: [Double], p0: [Double], peak: Double, air: Double?, hit: Double?, p1: simd_float3?)?
    private var lastPoseSent: TimeInterval = 0
    private var toastUntil: TimeInterval = 0

    static let trailSeconds: TimeInterval = 6
    static let syncSeconds: TimeInterval = 5
    /// The beacon's centre relative to the phone's camera at sync, in the phone's portrait axes (x right,
    /// y toward the top edge): beacon touching the phone's right edge, tops lined up.
    static let beaconOffset = simd_float2(0.045, -0.015)
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
        sceneView.scene.rootNode.addChildNode(pathRoot)
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
                endSync("SYNCED // ORIGIN SET")
            } else if msg["result"] as? String == "moving" {
                endSync("BEACON MOVED // SYNC AGAIN")
            } else {
                pendingCal = nil
            }
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

    func sync() {
        guard let frame = sceneView.session.currentFrame else { return }
        let pose = portraitPose(frame)
        guard pose.columns.2.y > 0.85 else { return show("LAY THE PHONE FLAT, SCREEN UP") }
        guard beaconLive else { return show("NO BEACON LINK") }
        let right = simd_make_float3(pose.columns.0), top = simd_make_float3(pose.columns.1)
        let origin = simd_make_float3(pose.columns.3) + right * Self.beaconOffset.x + top * Self.beaconOffset.y
        pendingCal = (origin, top)
        syncStarted = now
        calSent = false
        countdown = Int(Self.syncSeconds)
    }

    private func endSync(_ message: String) {
        pendingCal = nil
        syncStarted = nil
        countdown = nil
        show(message)
    }

    /// Runs on the overlay timer while syncing: abort if the beacon moves, and ask it to calibrate at the end
    /// of the window so it measures its biases while it's been sitting still the longest.
    private func syncStep() {
        guard let started = syncStarted else { return }
        let elapsed = now - started
        let beaconMoved = !beaconLive || (heartbeat["spin"] as? Double ?? 99) > 12 || (heartbeat["airborne"] as? Bool ?? false)
        if beaconMoved && !calSent { return endSync("BEACON MOVED // SYNC AGAIN") }
        // The beacon averages 0.8 s, so start it just in time to finish when the countdown does
        if !calSent && elapsed >= Self.syncSeconds - 0.8 {
            calSent = true
            link.send(["type": "cmd", "cmd": "cal"])
        }
        if elapsed > Self.syncSeconds + 3 { return endSync("NO REPLY FROM BEACON") }
        countdown = max(1, Int(ceil(Self.syncSeconds - elapsed)))
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

    // MARK: - Throw path

    /// Phone touching the bag where it'll be released. Throw within a couple of seconds after this.
    func markThrowStart() {
        guard let frame = sceneView.session.currentFrame else { return }
        if !airpod.running { airpod.start() }
        guard airpod.streaming else { return show("AIRPOD NOT STREAMING") }
        throwStart = (simd_make_float3(portraitPose(frame).columns.3), ProcessInfo.processInfo.systemUptime)
        pathRoot.childNodes.forEach { $0.removeFromParentNode() }
        pathText = "START MARKED // THROW, THEN TOUCH THE BAG AND TAP LANDED"
    }

    /// Phone touching the bag where it first hit. A thrown bag is ballistic, so the start point, this point and
    /// the airtime pin down the whole arc: p(t) = p0 + v t + ½ g t², with v solved from p(T) = p1.
    func markLanding() {
        guard let start = throwStart, let frame = sceneView.session.currentFrame else { return show("TAP THROW START FIRST") }
        // The first real flight after the start tap; later short ones are bounces
        guard let fall = airpod.falls.first(where: { $0.start > start.at && $0.airtime > 0.15 }) else {
            return show("NO THROW DETECTED")
        }
        let p0 = start.pos, p1 = simd_make_float3(portraitPose(frame).columns.3)
        let T = Float(fall.airtime)
        let g = simd_float3(0, -9.81, 0)  // ARKit world: Y up
        let v = (p1 - p0 - 0.5 * g * T * T) / T

        // Path frame for the drone: origin = release spot, Z up, Y = horizontal direction of the throw, X right
        let up = simd_float3(0, 1, 0)
        var fwd = simd_float3(p1.x - p0.x, 0, p1.z - p0.z)
        fwd = simd_length(fwd) > 0.01 ? simd_normalize(fwd) : simd_float3(0, 0, -1)
        let right = simd_cross(fwd, up)

        var points: [[Float]] = [], world: [[Float]] = []
        var last: simd_float3?
        let steps = max(2, Int(T * 30))
        for i in 0...steps {
            let t = T * Float(i) / Float(steps)
            let p = p0 + v * t + 0.5 * g * t * t
            let d = p - p0
            points.append([t, simd_dot(d, right), simd_dot(d, fwd), d.y])
            world.append([t, p.x, p.y, p.z])
            if let last { pathRoot.addChildNode(Self.segment(from: last, to: p, radius: 0.006, color: Self.yellow)) }
            last = p
        }
        let peak = points.map { $0[3] }.max() ?? 0
        let dist = simd_length(simd_float2(p1.x - p0.x, p1.z - p0.z))
        let vFwd = simd_dot(v, fwd)
        link.send(["type": "path", "airtime": T, "v0": [simd_dot(v, right), vFwd, v.y],
                   "frame": "origin = release spot, Z up, Y = throw direction, X right (m, s)",
                   "points": points, "world": world])
        pathText = String(format: "AIR %.2fs   DIST %.2fm   PEAK +%.2fm   V0 %.1f m/s @ %.0f°",
                          T, dist, peak, simd_length(v), atan2(v.y, vFwd) * 180 / .pi)
        throwStart = nil
    }

    // MARK: - Per frame

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
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
        syncStep()
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
