import CoreMotion
import Foundation

/// Experiment: can an AirPod's IMU stand in for the beacon as a free-fall stopwatch? Streams headphone motion,
/// reports the update rate and total acceleration, and times anything that reads near 0 g (in the air).
final class AirPodMotion: ObservableObject {
    @Published var status = "OFF"
    @Published var rateHz = 0.0
    @Published var accelG = 0.0
    @Published var lastAirtime: Double?
    @Published var running = false
    /// Every free fall seen, timed on the systemUptime clock: (when it started, how long it lasted)
    private(set) var falls: [(start: TimeInterval, airtime: TimeInterval)] = []
    var streaming: Bool { running && ProcessInfo.processInfo.systemUptime - lastSampleAt < 0.5 }

    private let manager = CMHeadphoneMotionManager()
    private var stamps: [TimeInterval] = []
    private var fallStart: TimeInterval?
    private var fallSamples = 0
    private var lastSampleAt: TimeInterval = 0
    private var watchdog: Timer?

    func toggle() { running ? stop() : start() }

    func start() {
        guard manager.isDeviceMotionAvailable else { status = "NOT SUPPORTED ON THESE AIRPODS"; return }
        running = true
        status = "WAITING FOR DATA (CONNECT AIRPODS)"
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self else { return }
            if let error { self.status = "ERROR: \(error.localizedDescription)"; return }
            guard let m = motion else { return }
            self.update(m)
        }
        // AirPods quietly stop sending when they think they're out of an ear
        watchdog = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self, self.running, !self.streaming else { return }
            self.status = "NO DATA (EAR DETECTION ON?)"
            self.rateHz = 0
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        watchdog?.invalidate()
        running = false
        status = "OFF"
        rateHz = 0
    }

    private func update(_ m: CMDeviceMotion) {
        let t = m.timestamp
        lastSampleAt = t
        stamps.append(t)
        stamps.removeAll { t - $0 > 1 }
        rateHz = Double(stamps.count)
        status = "STREAMING"

        // userAcceleration excludes gravity, so add it back: at rest this reads 1 g, in free fall about 0
        let a = m.userAcceleration, g = m.gravity
        accelG = sqrt(pow(a.x + g.x, 2) + pow(a.y + g.y, 2) + pow(a.z + g.z, 2))

        if accelG < 0.5 {
            if fallStart == nil { fallStart = t; fallSamples = 0 }
            fallSamples += 1
        } else if let start = fallStart {
            // Ignore single-sample dips; a real toss stays near 0 g for several samples
            if fallSamples >= 2 {
                lastAirtime = t - start
                falls.append((start, t - start))
                if falls.count > 20 { falls.removeFirst() }
            }
            fallStart = nil
        }
    }
}
