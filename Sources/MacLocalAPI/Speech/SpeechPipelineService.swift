import AVFoundation
import Foundation
import Speech

/// Orchestrator for the macOS 26 Speech pipeline.
///
/// Sequences AudioPreprocessor → ContextualVocabResolver →
/// SpeechTranscriberPool.checkout → SpeechTranscriberEngine.transcribe
/// → LanguageReassessor.shouldRetry, handling at most one speculative
/// retry with a detected locale.
///
/// Named `SpeechPipelineService` rather than `SpeechService` for now so it
/// can coexist with the legacy `Models/SpeechService.swift` during the
/// incremental migration. The final rename + controller wire-up +
/// legacy-deletion happens in a follow-up change once the live HTTP path
/// has been exercised end-to-end.
actor SpeechPipelineService {
    private let preprocessor: AudioPreprocessor
    private let resolver: ContextualVocabResolver
    private let pool: SpeechTranscriberPool
    private let reassessor: LanguageReassessor

    init(
        preprocessor: AudioPreprocessor,
        resolver: ContextualVocabResolver,
        pool: SpeechTranscriberPool,
        reassessor: LanguageReassessor
    ) {
        self.preprocessor = preprocessor
        self.resolver = resolver
        self.pool = pool
        self.reassessor = reassessor
    }

    /// Convenience constructor that wires up the default stack. Useful for
    /// test setup and for the HTTP server's startup path.
    static func makeDefault() async throws -> SpeechPipelineService {
        let preprocessor = AudioPreprocessor()
        let resolver = try ContextualVocabResolver()
        let pool = await SpeechTranscriberPool()
        return SpeechPipelineService(
            preprocessor: preprocessor,
            resolver: resolver,
            pool: pool,
            reassessor: LanguageReassessor()
        )
    }

    /// Transcribe an audio file. Applies the full pipeline — preprocessing,
    /// contextual vocab assembly, transcription, speculative language
    /// reassessment with optional one retry — and returns the final result.
    func transcribe(
        url: URL,
        options: PipelineRequestOptions
    ) async throws -> TranscriptionResult {
        // VAD trim is computed by AudioPreprocessor but currently does not
        // flow through to inference: SpeechAnalyzer reads the URL directly
        // here, so silence-trimming is decorative on this path. Two
        // attempted wire-ups in this branch did not pan out:
        //
        // 1) `start(inputSequence:)` with the prepared PCM chunks fed via
        //    AsyncStream<AnalyzerInput>: hard-crashed the process on
        //    first request with no useful diagnostics, likely an
        //    AVAudioPCMBuffer lifecycle race across the bridge from our
        //    chunk stream into the analyzer's input task.
        // 2) Materialize trimmed PCM to a temp WAV and pass the temp URL:
        //    works correctly but regresses median latency on the test
        //    corpus from ~240 ms to ~330 ms because every request pays
        //    AudioPreprocessor's resample + buffer-rewrite cost, while
        //    the say-synthesized fixtures have no leading silence to
        //    actually trim. A cheap pre-flight that sniffs the head of
        //    the file for silence and only invokes the full preprocessor
        //    when there is something to remove would flip the math, but
        //    that's its own change and out of scope here.
        //
        // Until one of those is sorted out we estimate the duration via
        // a one-shot AVAudioFile read and hand the original URL to the
        // engine.
        let durationSec = Self.audioDurationSeconds(url: url)
        let contextualStrings = resolver.resolve(prompt: options.prompt, locale: options.locale)

        let primaryLocale = Locale(identifier: options.locale)
        let engine = await pool.checkout(
            locale: primaryLocale,
            featureSet: .init(wantWordTimings: options.wantWordTimings)
        )

        let firstPass = try await engine.transcribe(
            url: url,
            locale: primaryLocale,
            contextualStrings: contextualStrings,
            wantWordTimings: options.wantWordTimings
        )

        var finalAttempt = firstPass
        var languageReassessed = false
        var finalLocaleIdentifier = options.locale

        if let retryLocale = reassessor.shouldRetry(
            attempt: firstPass,
            callerSuppliedLocale: options.callerSuppliedLocale,
            audioDurationSec: durationSec
        ) {
            let retryEngine = await pool.checkout(
                locale: retryLocale,
                featureSet: .init(wantWordTimings: options.wantWordTimings)
            )
            let retryVocab = resolver.resolve(prompt: options.prompt, locale: retryLocale.identifier)
            let retryAttempt = try await retryEngine.transcribe(
                url: url,
                locale: retryLocale,
                contextualStrings: retryVocab,
                wantWordTimings: options.wantWordTimings
            )
            finalAttempt = retryAttempt
            languageReassessed = true
            finalLocaleIdentifier = retryLocale.identifier
        }

        return TranscriptionResult(
            text: finalAttempt.text,
            language: finalLocaleIdentifier,
            durationMs: Int(durationSec * 1000),
            segments: finalAttempt.segments.isEmpty ? nil : finalAttempt.segments,
            words: finalAttempt.words.isEmpty ? nil : finalAttempt.words,
            languageReassessed: languageReassessed
        )
    }

    private static func audioDurationSeconds(url: URL) -> Double {
        guard let file = try? AVAudioFile(forReading: url) else { return 0 }
        let rate = file.processingFormat.sampleRate
        return rate > 0 ? Double(file.length) / rate : 0
    }
}

/// Request options shaped for the new pipeline. Kept here so the legacy
/// `SpeechRequestOptions` struct in `Models/SpeechService.swift` doesn't
/// need to be touched during incremental migration.
struct PipelineRequestOptions: Sendable {
    let locale: String
    let prompt: String?
    let wantWordTimings: Bool
    /// `true` if the caller supplied `language` explicitly — disables the
    /// speculative retry per spec Section 5.
    let callerSuppliedLocale: Bool

    init(
        locale: String = "en-US",
        prompt: String? = nil,
        wantWordTimings: Bool = true,
        callerSuppliedLocale: Bool = false
    ) {
        self.locale = locale
        self.prompt = prompt
        self.wantWordTimings = wantWordTimings
        self.callerSuppliedLocale = callerSuppliedLocale
    }
}
