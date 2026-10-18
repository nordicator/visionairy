import Foundation

/// WebSocket to server.js on the laptop (plain ws:// on port 8080). Delivers decoded JSON messages on the
/// main thread and reconnects on its own.
final class BeaconLink {
    var onMessage: (([String: Any]) -> Void)?
    private(set) var lastMessageAt: TimeInterval = 0

    private let url: URL
    private var task: URLSessionWebSocketTask?

    init(host: String) {
        url = URL(string: "ws://\(host):8080/ws")!
    }

    func connect() {
        task?.cancel()
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()
        receive()
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                if case .string(let text) = message,
                   let data = text.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    DispatchQueue.main.async {
                        self.lastMessageAt = Date().timeIntervalSince1970
                        self.onMessage?(obj)
                    }
                }
                self.receive()
            case .failure:
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.connect() }
            }
        }
    }

    func send(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { _ in }
    }

    var connected: Bool { Date().timeIntervalSince1970 - lastMessageAt < 2 }
}
