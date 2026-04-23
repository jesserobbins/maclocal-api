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
        // 1. Preprocess (we currently don't forward the streamed buffer into
        //    the engine; the engine opens the file directly and handles its
        //    own format matching. Preprocessing here is currently a no-op
        //    shape-preserving step until the engine learns to consume the
        //    streamed PCM buffer path in a follow-up. Duration is still
        //    useful for the reassessor.).
        let prepared = try await preprocessor.prepare(url: url)
        let durationSec = Double(prepared.durationMs) / 1000.0

        // 2. Resolve contextual strings.
        let contextualStrings = resolver.resolve(prompt: options.prompt, locale: options.locale)

        // 3. Check out engine for the primary locale.
        let primaryLocale = Locale(identifier: options.locale)
        let engine = await pool.checkout(
            locale: primaryLocale,
            featureSet: .init(wantWordTimings: options.wantWordTimings)
        )

        // 4. First-pass transcription.
        let firstPass = try await engine.transcribe(
            url: url,
            locale: primaryLocale,
            contextualStrings: contextualStrings,
            wantWordTimings: options.wantWordTimings
        )

        // 5. Reassess — should we retry under a different locale?
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
            // For the retry we re-resolve vocab against the new locale.
            // Currently the resolver is locale-agnostic, so this just returns
            // the same strings — but the hook is in place for future
            // locale-specific vocab bundles.
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

        // 6. Package result for the HTTP response.
        return TranscriptionResult(
            text: finalAttempt.text,
            language: finalLocaleIdentifier,
            durationMs: Int(durationSec * 1000),
            segments: finalAttempt.segments.isEmpty ? nil : finalAttempt.segments,
            words: finalAttempt.words.isEmpty ? nil : finalAttempt.words,
            languageReassessed: languageReassessed
        )
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
