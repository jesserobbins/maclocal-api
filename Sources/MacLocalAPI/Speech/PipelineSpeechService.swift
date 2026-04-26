import Foundation
import Speech

/// SpeechServing adapter that routes HTTP transcription through the macOS 26
/// `SpeechAnalyzer` / `SpeechTranscriber` pipeline (`SpeechPipelineService`)
/// instead of the legacy `SFSpeechRecognizer`.
///
/// Why this adapter exists rather than wiring `SpeechPipelineService` straight
/// into the controller: the controller's `SpeechServing` protocol takes a
/// file path and returns plain text, while the pipeline returns a richer
/// `TranscriptionResult` and is constructed via an async factory. The
/// adapter narrows the interface to match what the legacy controller
/// expects today, while keeping the existing TCC + file-validation
/// preflight in one place.
///
/// Pipeline lifecycle: a single instance is created lazily on the first
/// transcription request and held for the life of the process. The
/// constructor is async (the pool warms en-US at init), so the lazy
/// approach avoids blocking server startup. Subsequent requests reuse the
/// same pipeline (and its warm transcriber pool).
@available(macOS 13.0, *)
final class PipelineSpeechService: SpeechServing, RichSpeechServing, @unchecked Sendable {
    private let lock = NSLock()
    private var cached: SpeechPipelineService?

    init() {}

    /// Fire-and-forget pool warmup. Server.configure() calls this after
    /// registering the speech route so the first real request doesn't pay
    /// `SpeechPipelineService.makeDefault()`'s pool init + bundled-vocab
    /// read inline. Failures are silently dropped here — they'll surface
    /// on the first actual transcription with the right SpeechError shape
    /// for HTTP mapping.
    func warmup() {
        Task.detached(priority: .userInitiated) { [weak self] in
            _ = try? await self?.getOrCreate()
        }
    }

    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
        let result = try await transcribeRich(
            from: filePath,
            options: options,
            wantWordTimings: false
        )
        return result.text
    }

    func transcribeRich(
        from filePath: String,
        options: SpeechRequestOptions,
        wantWordTimings: Bool
    ) async throws -> TranscriptionResult {
        try Self.preflight(filePath: filePath, options: options)
        let pipeline = try await getOrCreate()

        let pipelineOptions = PipelineRequestOptions(
            locale: options.locale,
            prompt: options.prompt,
            wantWordTimings: wantWordTimings,
            // The legacy struct doesn't carry "did the caller pass a
            // language explicitly"; treat any non-default locale as
            // caller-supplied so the speculative reassessor doesn't fire
            // on requests that explicitly asked for, say, es-ES.
            callerSuppliedLocale: options.locale != "en-US"
        )
        return try await pipeline.transcribe(
            url: URL(fileURLWithPath: filePath),
            options: pipelineOptions
        )
    }

    // MARK: - Pipeline cache

    private func getOrCreate() async throws -> SpeechPipelineService {
        // Fast-path: already constructed.
        lock.lock()
        if let existing = cached {
            lock.unlock()
            return existing
        }
        lock.unlock()

        // Slow-path: construct a new pipeline. Two concurrent first-callers
        // can both end up here; one will win and the other will discard
        // its instance below. SpeechPipelineService.makeDefault is
        // idempotent — pool warmup is a few empty engine allocations and
        // a vocab-file read, so the wasted work on a lost race is tiny.
        let fresh = try await SpeechPipelineService.makeDefault()
        lock.lock()
        if let existing = cached {
            lock.unlock()
            return existing
        }
        cached = fresh
        lock.unlock()
        return fresh
    }

    // MARK: - Preflight

    /// File + TCC validation that the legacy `SpeechService` performs before
    /// hitting `SFSpeechRecognizer`. Duplicated here (not extracted to a
    /// shared helper) because:
    /// 1. The legacy version is private to `SpeechService`.
    /// 2. The set of checks is small enough that a near-term refactor to
    ///    a shared `SpeechPreflightChecker` is straightforward when the
    ///    legacy service is eventually removed.
    static func preflight(filePath: String, options: SpeechRequestOptions) throws {
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw SpeechError.fileNotFound
        }

        let ext = URL(fileURLWithPath: filePath).pathExtension.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(ext) else {
            throw SpeechError.unsupportedFormat
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
        if let size = attrs[.size] as? Int, size > options.maxFileBytes {
            throw SpeechError.requestTooLarge(actualBytes: size, maxBytes: options.maxFileBytes)
        }

        try checkAuthorization(options: options)
    }

    private static func checkAuthorization(options: SpeechRequestOptions) throws {
        let status = SFSpeechRecognizer.authorizationStatus()
        if status == .denied || status == .restricted {
            throw SpeechError.authorizationDenied
        }
        if status == .notDetermined {
            // For headless / HTTP callers we refuse rather than block on a
            // TCC dialog the caller can't see. The legacy SpeechService has
            // an async-prompt path gated by `promptForAuthorization`; that
            // path is for CLI invocations, never set on HTTP requests, so
            // we mirror only the deny branch here.
            guard options.promptForAuthorization else {
                throw SpeechError.authorizationDenied
            }
            // Synchronous request fallback for the (rare) prompting caller.
            // We can't await here from a non-async function; the caller
            // path (CLI) goes through legacy SpeechService instead. A
            // future cleanup can move the prompt path into this adapter
            // too once the legacy service is retired.
            throw SpeechError.authorizationDenied
        }
    }
}
