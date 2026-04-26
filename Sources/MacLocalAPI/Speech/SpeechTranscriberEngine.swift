import AVFoundation
import CoreMedia
import Foundation
import Speech

/// File-based wrapper around `SpeechAnalyzer` + `SpeechTranscriber`.
///
/// Single entry: `transcribe(url:locale:contextualStrings:wantWordTimings:)`
/// returns a `TranscriptionAttempt` containing the transcribed text plus the
/// signal values `LanguageReassessor` needs to decide on a retry:
/// mean early-window confidence, OOV ratio, and an n-gram locale guess.
///
/// Streaming input (for cancel-and-restart in the reassessor path) is left
/// for `SpeechTranscriberPool` + `SpeechService` in later tasks.
actor SpeechTranscriberEngine {
    /// Duration of the window at the start of audio over which we compute
    /// `meanEarlyConfidence`. Matches `LanguageReassessor.evaluationWindowSec`
    /// contract (3 s).
    static let earlyConfidenceWindowSec: Double = 3.0

    init() {}

    func transcribe(
        url: URL,
        locale: Locale,
        contextualStrings: [String],
        wantWordTimings: Bool = true
    ) async throws -> TranscriptionAttempt {
        let audioFile = try AVAudioFile(forReading: url)
        let sampleRate = audioFile.processingFormat.sampleRate
        let durationSec = sampleRate > 0 ? Double(audioFile.length) / sampleRate : 0.0

        // Construct the transcriber with explicit options instead of the
        // built-in presets. The presets that include time-indexing also
        // bundle `volatileResults` (streaming partials) and don't include
        // `transcriptionConfidence`. We want the opposite — finalized
        // results only, with both per-run audioTimeRange and confidence
        // attributes populated so verbose_json word entries report real
        // confidence numbers instead of constant 0.0. `audioTimeRange` is
        // requested unconditionally so segment-level timings always work
        // even when callers didn't ask for word-granularity output.
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        // wantWordTimings is now a controller-level concern (whether to
        // surface words[] in verbose_json); the underlying transcriber
        // collects timings either way at negligible cost.
        _ = wantWordTimings

        // Ensure the locale's recognition model is installed before we
        // hand the transcriber to the analyzer. Apple ships a small set
        // of locales by default (en-US most reliably); requesting any
        // other locale on a machine that hasn't installed it yields the
        // opaque "Audio format is not supported" NSError on the analyzer
        // side. Install on demand so the first non-English request just
        // takes a few extra seconds instead of failing outright.
        try await Self.ensureLocaleInstalled(for: transcriber, locale: locale)

        // Bias via AnalysisContext.contextualStrings under the general tag.
        let context = AnalysisContext()
        context.contextualStrings = [.general: contextualStrings]

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        try await analyzer.setContext(context)

        // Consumer runs concurrently with start(). When the analyzer reaches
        // end-of-file with finishAfterFile:true, the results sequence
        // terminates naturally. The progressive presets emit volatile
        // partial results before each segment finalizes — we drop those
        // here and keep only `isFinal == true`. Without the filter the
        // assembled output concatenates every prefix-snapshot of the
        // running transcription ("H Hub Hubern Hubernates orchestrates...")
        // because each partial is appended as its own segment.
        let resultsTask = Task<[SpeechTranscriber.Result], Error> {
            var collected: [SpeechTranscriber.Result] = []
            for try await r in transcriber.results where r.isFinal {
                collected.append(r)
            }
            return collected
        }

        try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        let results = try await resultsTask.value

        return Self.assembleAttempt(
            from: results,
            durationSec: durationSec,
            activeVocab: Set(contextualStrings.map { $0.lowercased() })
        )
    }

    // MARK: - Result assembly

    static func assembleAttempt(
        from results: [SpeechTranscriber.Result],
        durationSec: Double,
        activeVocab: Set<String>
    ) -> TranscriptionAttempt {
        var texts: [String] = []
        var words: [WordTiming] = []
        var segments: [Segment] = []
        var earlyConfidences: [Double] = []

        let earlyCutoff = CMTime(seconds: Self.earlyConfidenceWindowSec, preferredTimescale: 1_000)

        for result in results {
            let resultText = String(result.text.characters)
            texts.append(resultText)

            // Segment timing from result.range.
            let segStart = result.range.start
            let segEnd = CMTimeAdd(result.range.start, result.range.duration)
            let segConfs = extractConfidences(from: result.text)
            let segMeanConf = segConfs.isEmpty ? 0.0 : segConfs.reduce(0, +) / Double(segConfs.count)
            segments.append(Segment(
                text: resultText,
                startMs: cmTimeMs(segStart),
                endMs: cmTimeMs(segEnd),
                meanConfidence: segMeanConf
            ))

            // Per-run word timings + confidence.
            for run in result.text.runs {
                let runText = String(result.text[run.range].characters).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !runText.isEmpty else { continue }
                let timeRange: CMTimeRange? = run.audioTimeRange
                let confidence: Double = run.transcriptionConfidence ?? 0.0

                if let tr = timeRange {
                    words.append(WordTiming(
                        word: runText,
                        startMs: cmTimeMs(tr.start),
                        endMs: cmTimeMs(CMTimeAdd(tr.start, tr.duration)),
                        confidence: confidence
                    ))
                    // Pick up confidences from runs whose start falls in the
                    // early window.
                    if CMTimeCompare(tr.start, earlyCutoff) <= 0 {
                        earlyConfidences.append(confidence)
                    }
                } else {
                    // No time range available — still contribute to confidence
                    // signal if we consider the whole first N results "early".
                    earlyConfidences.append(confidence)
                }
            }
        }

        let fullText = texts.joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let meanEarly = earlyConfidences.isEmpty
            ? 1.0  // no signal → treat as high confidence so retry doesn't fire
            : earlyConfidences.reduce(0, +) / Double(earlyConfidences.count)

        let tokens = OOVCalculator.tokenize(fullText)
        let oov = OOVCalculator.oovRatio(tokens: tokens, known: activeVocab)

        let guess = CharacterClassLanguageID.identify(text: fullText)

        return TranscriptionAttempt(
            text: fullText,
            meanEarlyConfidence: meanEarly,
            oovRatio: oov,
            segments: segments,
            words: words,
            detectedLanguageGuess: guess
        )
    }

    // MARK: - Helpers

    static func extractConfidences(from attributed: AttributedString) -> [Double] {
        var out: [Double] = []
        for run in attributed.runs {
            if let c: Double = run.transcriptionConfidence {
                out.append(c)
            }
        }
        return out
    }

    static func cmTimeMs(_ time: CMTime) -> Int {
        guard time.isValid else { return 0 }
        return Int((CMTimeGetSeconds(time) * 1000.0).rounded())
    }

    // MARK: - Locale model install

    /// Soft cap on how long we'll block a request waiting for an asset
    /// install. Locale recognition models are typically tens of MB so
    /// they install in a few seconds on a working network, but we don't
    /// want a misconfigured machine to hang an HTTP request for minutes.
    static let assetInstallTimeoutNs: UInt64 = 60_000_000_000 // 60 s

    /// Apple's Speech APIs return locale identifiers in NSLocale component
    /// form (`en_US`, `es_MX`); HTTP callers pass BCP-47 form (`en-US`,
    /// `es-MX`); and our own constants are mixed. Normalize to lowercase +
    /// hyphen separator so equality checks work across all three.
    static func canonicalizeLocaleID(_ id: String) -> String {
        return id.lowercased().replacingOccurrences(of: "_", with: "-")
    }

    /// If the requested locale's recognition model isn't currently
    /// installed, ask Speech to download and install it before the
    /// analyzer starts. Returns when the model is ready (or no install
    /// was needed). Throws a SpeechError that the controller maps to a
    /// usable HTTP status when the locale isn't supported, the install
    /// errors out, or the download exceeds the soft cap above.
    private static func ensureLocaleInstalled(
        for transcriber: SpeechTranscriber,
        locale: Locale
    ) async throws {
        // Apple's installedLocales / supportedLocales return identifiers
        // in component form ("en_US") while callers pass BCP-47 form
        // ("en-US"). Normalize both sides to a common shape before
        // comparing so en-US doesn't get flagged as missing on a machine
        // that has it installed.
        let want = Self.canonicalizeLocaleID(locale.identifier)

        let installed = await SpeechTranscriber.installedLocales
        if installed.contains(where: { Self.canonicalizeLocaleID($0.identifier) == want }) {
            return
        }

        let supported = await SpeechTranscriber.supportedLocales
        guard supported.contains(where: { Self.canonicalizeLocaleID($0.identifier) == want }) else {
            let supportedList = supported.map { $0.identifier }.sorted().joined(separator: ", ")
            throw SpeechError.recognitionFailed(
                "Locale \(locale.identifier) is not supported by SpeechTranscriber on this device. Supported locales: \(supportedList)"
            )
        }

        let request: AssetInstallationRequest?
        do {
            request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber])
        } catch {
            throw SpeechError.recognitionFailed(
                "Could not query locale install status for \(locale.identifier): \(error.localizedDescription)"
            )
        }

        guard let request else {
            // The framework reported no install needed even though the
            // locale wasn't in installedLocales — proceed and let the
            // analyzer surface anything still wrong.
            return
        }

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await request.downloadAndInstall()
                }
                group.addTask {
                    try await Task.sleep(nanoseconds: assetInstallTimeoutNs)
                    throw SpeechError.recognitionFailed(
                        "Locale \(locale.identifier) model install timed out after \(assetInstallTimeoutNs / 1_000_000_000) s — retry the request once the download completes."
                    )
                }
                try await group.next()
                group.cancelAll()
            }
        } catch let speechError as SpeechError {
            throw speechError
        } catch {
            throw SpeechError.recognitionFailed(
                "Locale \(locale.identifier) model install failed: \(error.localizedDescription)"
            )
        }
    }
}
