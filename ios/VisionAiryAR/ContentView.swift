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
                    Text("CALIBRATING IN \(n)").font(.system(.title, design: .monospaced).bold())
                    Text("HOLD STILL").font(.system(.caption, design: .monospaced))
                } else if tracker.phase == "UNCAL" {
                    Text("To calibrate: hold the phone flat, screen up, at waist height, and lay the beacon on the screen with its Y arrow pointing to the top of the phone. Hold still for 2 seconds.")
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

                HStack(spacing: 10) {
                    button("CALIBRATE") { tracker.calibrateNow() }
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

    private func row(_ label: String, _ value: String, _ color: Color) -> some View {
        HStack {
            Text(label).opacity(0.7)
            Spacer()
            Text(value).foregroundColor(color)
        }
    }

    private func button(_ title: String, action: @escaping () -> Void) -> some View {
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

struct ARContainer: UIViewRepresentable {
    let view: ARSCNView
    func makeUIView(context: Context) -> ARSCNView { view }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
