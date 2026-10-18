import ARKit
import Vision
import simd

/// Finds the drone in 3D with the camera and LiDAR. Tap the drone on screen once: Vision follows it from
/// frame to frame, and the LiDAR depth inside its box gives the distance, so its position comes out in
/// ARKit's world coordinates (the same ones the throw path is drawn in).
final class DroneTracker {
    /// Latest world position (ARKit world: Y up, metres), or nil for a frame without one; `lost` means Vision
    /// lost the drone and it needs tapping again. Called on the main thread.
    var onUpdate: ((_ position: simd_float3?, _ lost: Bool) -> Void)?

    static var supported: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    /// Rough size of the drone, used to size the box Vision follows at whatever distance it was tapped
    static let droneSize: Float = 0.15

    private struct Snapshot {
        let image: CVPixelBuffer
        let depth: CVPixelBuffer
        let confidence: CVPixelBuffer?
        let camera: simd_float4x4
        let intrinsics: simd_float3x3
        let size: simd_float2   // captured image size in pixels
    }

    private let queue = DispatchQueue(label: "drone-tracker")
    private let sequence = VNSequenceRequestHandler()
    private var request: VNTrackObjectRequest?
    private var pendingTap: CGPoint?   // normalized image coords, top-left origin
    private var lastDepth: Float?
    private var busy = false
    private(set) var active = false

    /// Point (in normalized captured-image coordinates, top-left origin) where the drone was tapped
    func select(_ point: CGPoint) {
        queue.async {
            self.pendingTap = point
            self.request = nil
            self.lastDepth = nil
        }
        active = true
    }

    func stop() {
        active = false
        queue.async {
            self.pendingTap = nil
            self.request = nil
        }
        onUpdate?(nil, false)
    }

    /// Feed every AR frame; frames that arrive while the previous one is still being processed are skipped
    func process(_ frame: ARFrame) {
        guard active, !busy, let depth = frame.sceneDepth else { return }
        let res = frame.camera.imageResolution
        let snap = Snapshot(image: frame.capturedImage, depth: depth.depthMap, confidence: depth.confidenceMap,
                            camera: frame.camera.transform, intrinsics: frame.camera.intrinsics,
                            size: simd_float2(Float(res.width), Float(res.height)))
        busy = true
        queue.async {
            let pos = self.track(snap)
            let lost = self.request == nil && self.pendingTap == nil
            DispatchQueue.main.async {
                self.busy = false
                if self.active { self.onUpdate?(pos, lost) }
            }
        }
    }

    private func track(_ s: Snapshot) -> simd_float3? {
        if let tap = pendingTap {
            pendingTap = nil
            // Box about the drone's size at the depth where it was tapped
            guard let d = depth(at: tap, radius: 3, s) else { return nil }
            let px = s.intrinsics[0][0] * Self.droneSize / d
            let w = CGFloat(px / s.size.x), h = CGFloat(px / s.size.y)
            let box = CGRect(x: tap.x - w / 2, y: 1 - tap.y - h / 2, width: w, height: h)   // Vision: bottom-left origin
            let r = VNTrackObjectRequest(detectedObjectObservation: VNDetectedObjectObservation(boundingBox: box))
            r.trackingLevel = .accurate
            request = r
            lastDepth = d
        }
        guard let r = request else { return nil }
        do { try sequence.perform([r], on: s.image, orientation: .up) } catch { request = nil; return nil }
        guard let obs = r.results?.first as? VNDetectedObjectObservation, obs.confidence > 0.3 else {
            request = nil
            return nil
        }
        r.inputObservation = obs
        let centre = CGPoint(x: obs.boundingBox.midX, y: 1 - obs.boundingBox.midY)
        let radius = max(1, Int(obs.boundingBox.width * CGFloat(CVPixelBufferGetWidth(s.depth)) / 4))
        guard let d = depth(at: centre, radius: radius, s) else { return nil }
        // A jump this big in one frame is the background showing through, not the drone moving
        if let last = lastDepth, abs(d - last) > 0.4 { return nil }
        lastDepth = d

        // Pixel + depth -> camera space (x right, y up, looking down -z) -> world
        let u = Float(centre.x) * s.size.x, v = Float(centre.y) * s.size.y
        let k = s.intrinsics
        let local = simd_float4((u - k[2][0]) / k[0][0] * d, -(v - k[2][1]) / k[1][1] * d, -d, 1)
        return simd_make_float3(s.camera * local)
    }

    /// Depth of the drone around a point: the nearer readings in a small window, so the wall behind it
    /// doesn't count. Low-confidence LiDAR pixels are skipped.
    private func depth(at p: CGPoint, radius: Int, _ s: Snapshot) -> Float? {
        let map = s.depth
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        if let c = s.confidence { CVPixelBufferLockBaseAddress(c, .readOnly) }
        defer { if let c = s.confidence { CVPixelBufferUnlockBaseAddress(c, .readOnly) } }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let row = CVPixelBufferGetBytesPerRow(map)
        let cx = min(max(Int(p.x * CGFloat(w)), 0), w - 1), cy = min(max(Int(p.y * CGFloat(h)), 0), h - 1)
        var values: [Float] = []
        for y in max(0, cy - radius)...min(h - 1, cy + radius) {
            let line = base.advanced(by: y * row).assumingMemoryBound(to: Float32.self)
            for x in max(0, cx - radius)...min(w - 1, cx + radius) {
                if let c = s.confidence, let cb = CVPixelBufferGetBaseAddress(c) {
                    let conf = cb.advanced(by: y * CVPixelBufferGetBytesPerRow(c)).assumingMemoryBound(to: UInt8.self)[x]
                    if conf < UInt8(ARConfidenceLevel.medium.rawValue) { continue }
                }
                let d = line[x]
                if d > 0.1 && d < 6 { values.append(d) }
            }
        }
        guard !values.isEmpty else { return nil }
        values.sort()
        return values[values.count / 4]
    }
}
