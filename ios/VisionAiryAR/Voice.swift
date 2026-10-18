import AVFoundation
import Speech

/// Live speech-to-text, on the phone itself (no internet needed, which matters on the drone's Wi-Fi).
/// Words appear while you're still talking, and spoken commands fire as soon as they're heard.
final class Voice: ObservableObject {
    @Published var listening = false
    @Published var text = ""
    @Published var status = ""

    /// A command word was heard (see `Voice.commands`). Called on the main thread.
    var onCommand: ((String) -> Void)?
    /// Each finished phrase. Called on the main thread.
    var onPhrase: ((String) -> Void)?

    /// Spoken phrase -> command. Whole words only, so "landed" never counts as "land".
    static let commands: [(words: [String], command: String)] = [
        (["throw", "start"], "throw start"), (["start", "throw"], "throw start"),
        (["landed"], "landed"),
        (["sync"], "sync"), (["re", "zero"], "rezero"), (["rezero"], "rezero"), (["clear"], "clear"),
        (["find", "drone"], "find drone"), (["track", "drone"], "find drone"),
        (["follow"], "follow"), (["land"], "land"),
        (["stop"], "stop"), (["cancel"], "stop"), (["hover"], "stop"),
    ]

    private let engine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var consumed = 0   // words of the current phrase already checked for commands

    func toggle() { listening ? stop() : start() }

    func start() {
        SFSpeechRecognizer.requestAuthorization { auth in
            AVAudioSession.sharedInstance().requestRecordPermission { mic in
                DispatchQueue.main.async {
                    guard auth == .authorized, mic else { self.status = "ALLOW MIC + SPEECH IN SETTINGS"; return }
                    self.begin()
                }
            }
        }
    }

    func stop() {
        listening = false
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        status = ""
    }

    private func begin() {
        guard let recognizer, recognizer.isAvailable else { status = "SPEECH UNAVAILABLE"; return }
        do {
            let session = AVAudioSession.sharedInstance()
            // The phone's own mic (no Bluetooth), so it doesn't fight the AirPod motion stream
            try session.setCategory(.record, mode: .measurement, options: [.mixWithOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            let input = engine.inputNode
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { [weak self] buf, _ in
                self?.request?.append(buf)
            }
            engine.prepare()
            try engine.start()
        } catch {
            status = "MIC ERROR"
            return
        }
        listening = true
        status = recognizer.supportsOnDeviceRecognition ? "ON DEVICE" : "NEEDS INTERNET"
        newPhrase()
    }

    /// One recognition task per phrase; a new one starts as soon as the last finishes, so listening never stops
    private func newPhrase() {
        guard listening, let recognizer else { return }
        let r = SFSpeechAudioBufferRecognitionRequest()
        r.shouldReportPartialResults = true
        r.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        r.addsPunctuation = true
        r.taskHint = .dictation
        r.contextualStrings = ["re-zero", "sync", "throw start", "landed", "drone"]
        request = r
        consumed = 0
        task = recognizer.recognitionTask(with: r) { [weak self] result, error in
            DispatchQueue.main.async { self?.handle(result, error, r) }
        }
    }

    private func handle(_ result: SFSpeechRecognitionResult?, _ error: Error?, _ r: SFSpeechAudioBufferRecognitionRequest) {
        guard r === request else { return }   // a finished phrase's late callback
        if let result {
            let phrase = result.bestTranscription.formattedString
            text = phrase
            checkCommands(phrase)
            if result.isFinal {
                if !phrase.isEmpty { onPhrase?(phrase) }
                return newPhrase()
            }
        }
        if error != nil { newPhrase() }
    }

    /// Fire commands in the words heard since the last check. Partial results only grow, so each word is
    /// looked at once.
    private func checkCommands(_ phrase: String) {
        let words = phrase.lowercased().split { !$0.isLetter }.map(String.init)
        guard words.count > consumed else { return }
        var i = max(0, consumed - 1)   // a two-word command can straddle the last check
        while i < words.count {
            if let match = Self.commands.first(where: { c in
                i + c.words.count <= words.count && Array(words[i..<i + c.words.count]) == c.words
            }) {
                onCommand?(match.command)
                i += match.words.count
                consumed = i
            } else {
                i += 1
            }
        }
        // The last word may still grow ("land" -> "landed"), so leave it for the next check
        consumed = max(consumed, words.count - 1)
    }
}
