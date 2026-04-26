import ArgumentParser
import AVFoundation
import CoreMedia
import Foundation
import Speech

/// `afm speech-custom-lm-test -f FILE` — spike command that exercises the
/// SFCustomLanguageModelData → SFSpeechLanguageModel → DictationTranscriber
/// path end-to-end on a single audio file. Prints both the standard
/// SpeechTranscriber output and the DictationTranscriber+custom-LM output
/// so they can be compared side-by-side.
///
/// Not a long-lived feature — the goal is to validate the custom-LM API
/// works on this machine before committing to a pipeline rewrite. If
/// DictationTranscriber recovers stubborn product names like "PostgreSQL"
/// where SpeechTranscriber didn't, that justifies wiring it into the
/// HTTP path; if not, the CLM idea is documented as tried-and-bounded.
@available(macOS 26.0, *)
struct SpeechCustomLMTestCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech-custom-lm-test",
        abstract: "Compare SpeechTranscriber vs DictationTranscriber+custom-LM on one audio file"
    )

    @ArgumentParser.Option(name: [.customShort("f"), .long], help: "Path to a 16 kHz mono WAV / supported audio file")
    var file: String

    @ArgumentParser.Option(name: .long, help: "Locale to transcribe in")
    var locale: String = "en-US"

    func run() throws {
        // Bridge sync ArgumentParser entrypoint to our async work via a
        // CheckedContinuation-style wait. Could move to AsyncParsable
        // root, but that change cascades through main.swift's other
        // entry points; this command is a one-off spike anyway.
        let group = DispatchGroup()
        group.enter()
        var caught: Error?
        Task {
            do {
                try await runAsync()
            } catch {
                caught = error
            }
            group.leave()
        }
        group.wait()
        if let caught {
            throw caught
        }
    }

    func runAsync() async throws {
        let url = URL(fileURLWithPath: NSString(string: file).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("ERROR: file not found: \(url.path)")
            throw ExitCode.failure
        }

        let resolvedLocale = Locale(identifier: locale)

        // 1. Build + compile the custom language model. Time the steps
        //    so future tuning has a baseline for "how long does CLM init
        //    cost on a fresh machine".
        print("Step 1: Build + compile custom language model …")
        let manager = CustomLanguageModelManager()
        let compileStart = Date()
        let prepared: CustomLanguageModelManager.PreparedModel
        do {
            prepared = try await manager.prepare(locale: resolvedLocale)
        } catch {
            print("  ❌ CLM prepare failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
        let compileMs = Date().timeIntervalSince(compileStart) * 1000
        print(String(format: "  ✅ CLM ready in %.0f ms", compileMs))
        print("     model:      \(prepared.configuration.languageModel.path)")
        if let vocabURL = prepared.configuration.vocabulary {
            print("     vocabulary: \(vocabURL.path)")
        }
        print("")

        // 2. Baseline — SpeechTranscriber + bundled contextualStrings.
        print("Step 2: SpeechTranscriber baseline (current HTTP-path engine) …")
        let baselineStart = Date()
        let baseline: String
        do {
            baseline = try await Self.transcribeWithSpeech(url: url, locale: resolvedLocale)
        } catch {
            print("  ❌ baseline failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
        let baselineMs = Date().timeIntervalSince(baselineStart) * 1000
        print(String(format: "  ⏱  %.0f ms", baselineMs))
        print("  📝 \(baseline)")
        print("")

        // 3. DictationTranscriber + customizedLanguage(modelConfiguration:).
        print("Step 3: DictationTranscriber + custom LM …")
        let dictationStart = Date()
        let dictation: String
        do {
            dictation = try await Self.transcribeWithDictation(
                url: url,
                locale: resolvedLocale,
                modelConfiguration: prepared.configuration
            )
        } catch {
            print("  ❌ dictation failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
        let dictationMs = Date().timeIntervalSince(dictationStart) * 1000
        print(String(format: "  ⏱  %.0f ms", dictationMs))
        print("  📝 \(dictation)")
        print("")

        // 4. Side-by-side word-level diff so the impact (or lack of it)
        //    on stubborn terms is obvious.
        let baselineWords = baseline.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
        let dictationWords = dictation.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
        let baselineSet = Set(baselineWords)
        let dictationSet = Set(dictationWords)
        let recovered = dictationSet.subtracting(baselineSet).sorted()
        let lost = baselineSet.subtracting(dictationSet).sorted()
        if !recovered.isEmpty {
            print("Words recovered by DictationTranscriber+CLM:")
            print("  + " + recovered.joined(separator: ", "))
        }
        if !lost.isEmpty {
            print("Words present in baseline but absent in dictation:")
            print("  - " + lost.joined(separator: ", "))
        }
    }

    // MARK: - SpeechTranscriber path

    static func transcribeWithSpeech(url: URL, locale: Locale) async throws -> String {
        let audioFile = try AVAudioFile(forReading: url)
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let resultsTask = Task<[SpeechTranscriber.Result], Error> {
            var collected: [SpeechTranscriber.Result] = []
            for try await r in transcriber.results where r.isFinal {
                collected.append(r)
            }
            return collected
        }
        try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        let results = try await resultsTask.value
        return results.map { String($0.text.characters) }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - DictationTranscriber path

    static func transcribeWithDictation(
        url: URL,
        locale: Locale,
        modelConfiguration: SFSpeechLanguageModel.Configuration
    ) async throws -> String {
        let audioFile = try AVAudioFile(forReading: url)
        let transcriber = DictationTranscriber(
            locale: locale,
            contentHints: [.customizedLanguage(modelConfiguration: modelConfiguration)],
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let resultsTask = Task<[DictationTranscriber.Result], Error> {
            var collected: [DictationTranscriber.Result] = []
            for try await r in transcriber.results where r.isFinal {
                collected.append(r)
            }
            return collected
        }
        try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        let results = try await resultsTask.value
        return results.map { String($0.text.characters) }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
