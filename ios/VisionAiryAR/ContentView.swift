import ARKit
import SwiftUI

@main
struct VisionAiryARApp: App {
    var body: some Scene {
        WindowGroup { ContentView().preferredColorScheme(.dark) }
    }
}

private let hud = Color(red: 0.22, green: 1, blue: 0.42)
private let pathYellow = Color(red: 1, green: 0.82, blue: 0.25)

struct ContentView: View {
    @StateObject private var tracker = Tracker()

    var body: some View {
        ZStack {
            ARContainer(view: tracker.sceneView).ignoresSafeArea()

            VStack(spacing: 8) {
                // Status
                HStack {
                    Text(tracker.linkText)
                    Spacer()
                    Text(tracker.phase).foregroundColor(tracker.phase == "FLIGHT" ? pathYellow : tracker.phase == "UNCAL" ? .red : hud)
                }
                .font(.system(.subheadline, design: .monospaced).bold())
                if !tracker.trackingText.isEmpty {
                    Text(tracker.trackingText).font(.system(.caption, design: .monospaced)).foregroundColor(.orange)
                }

                Spacer()

                if let n = tracker.countdown {
                    Text("SYNCING \(n)").font(.system(.title, design: .monospaced).bold())
                    Text("DON'T TOUCH THE BEACON // PHONE CAN MOVE").font(.system(.caption, design: .monospaced))
                } else if tracker.phase == "UNCAL" && !tracker.airpod.running {
                    Text("To sync: lay the phone flat, screen up, with the beacon touching its right edge, tops lined up and the Y arrow pointing the same way as the top of the phone. Tap SYNC, then leave the beacon alone for 5 seconds.")
                        .font(.system(.footnote, design: .monospaced))
                        .multilineTextAlignment(.center)
                        .padding(10)
                        .background(.black.opacity(0.6))
                }
                if !tracker.toast.isEmpty {
                    Text(tracker.toast).font(.system(.headline, design: .monospaced).bold())
                }

                Spacer()

                // Coordinates: origin = calibration spot, X right, Y along the beacon's arrow, Z up
                VStack(alignment: .leading, spacing: 3) {
                    row("BEACON", tracker.beaconText, pathYellow)
                    row("CAMERA", tracker.camText, Color(red: 0.42, green: 0.72, blue: 1))
                    row("DISTANCE", tracker.distText, hud)
                    if !tracker.throwText.isEmpty {
                        Text(tracker.throwText).foregroundColor(pathYellow).lineLimit(2).minimumScaleFactor(0.7)
                    }
                }
                .font(.system(.caption, design: .monospaced))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.black.opacity(0.55))

                AirPodPanel(airpod: tracker.airpod)
                if !tracker.pathText.isEmpty {
                    Text(tracker.pathText).font(.system(.caption, design: .monospaced)).foregroundColor(pathYellow)
                        .lineLimit(2).minimumScaleFactor(0.7)
                }

                HStack(spacing: 10) {
                    button("THROW START") { tracker.markThrowStart() }
                    button("LANDED") { tracker.markLanding() }
                }
                HStack(spacing: 10) {
                    button("SYNC") { tracker.sync() }
                    button("RE-ZERO") { tracker.rezero() }
                    button("CLEAR") { tracker.clearTrail() }
                }
                Text("Re-zero: touch the beacon to the phone, then tap.").font(.system(.caption2, design: .monospaced)).opacity(0.7)
            }
            .foregroundColor(hud)
            .shadow(color: .black, radius: 2)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .onAppear { tracker.start() }
    }

    func row(_ label: String, _ value: String, _ color: Color) -> some View {
        HStack {
            Text(label).opacity(0.7)
            Spacer()
            Text(value).foregroundColor(color)
        }
    }

    func button(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(.footnote, design: .monospaced).bold())
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(hud, lineWidth: 1.5))
        }
        .background(.black.opacity(0.5))
    }
}

/// AirPod motion readout (its own view so it refreshes on the AirPod's updates)
struct AirPodPanel: View {
    @ObservedObject var airpod: AirPodMotion

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("AIRPOD").opacity(0.7)
                Spacer()
                Text(airpod.status)
                Button(airpod.running ? "OFF" : "ON") { airpod.toggle() }.padding(.leading, 8)
            }
            if airpod.running {
                HStack { Text("RATE / ACCEL").opacity(0.7); Spacer(); Text(String(format: "%.0f Hz   %.2f g", airpod.rateHz, airpod.accelG)) }
                HStack { Text("LAST AIRTIME").opacity(0.7); Spacer(); Text(airpod.lastAirtime.map { String(format: "%.3f s", $0) } ?? "--") }
            }
        }
        .font(.system(.caption, design: .monospaced))
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.55))
    }
}

struct ARContainer: UIViewRepresentable {
    let view: ARSCNView
    func makeUIView(context: Context) -> ARSCNView { view }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
