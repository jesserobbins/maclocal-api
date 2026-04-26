import Vapor
import Foundation

struct SpeechTranscriptionResponse: Content {
    let object: String
    let text: String
    let locale: String
}

/// OpenAI-compatible verbose_json shape. Fields beyond `text` are present
/// when the underlying transcriber produced them (i.e. the new
/// SpeechAnalyzer pipeline path); on the legacy SFSpeechRecognizer fallback
/// the segments / words arrays are nil and `language_reassessed` is false.
struct VerboseTranscriptionResponse: Content {
    let task: String
    let language: String
    let duration: Double
    let text: String
    let segments: [VerboseSegment]?
    let words: [VerboseWord]?
    let languageReassessed: Bool

    enum CodingKeys: String, CodingKey {
        case task, language, duration, text, segments, words
        case languageReassessed = "language_reassessed"
    }
}

struct VerboseSegment: Content {
    let id: Int
    let start: Double
    let end: Double
    let text: String
    let avgConfidence: Double

    enum CodingKeys: String, CodingKey {
        case id, start, end, text
        case avgConfidence = "avg_confidence"
    }
}

struct VerboseWord: Content {
    let word: String
    let start: Double
    let end: Double
    let confidence: Double
}

protocol SpeechServing {
    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String
}

/// Extended interface for backends that can return word/segment timings and
/// other rich transcription metadata. The HTTP controller queries for this
/// conformance when `response_format=verbose_json`. Legacy
/// `SFSpeechRecognizer`-backed `SpeechService` does not conform — it would
/// have to do a second pass to recover word timings and the live HTTP path
/// has been moved off it anyway.
protocol RichSpeechServing: SpeechServing {
    func transcribeRich(
        from filePath: String,
        options: SpeechRequestOptions,
        wantWordTimings: Bool
    ) async throws -> TranscriptionResult
}

extension SpeechService: SpeechServing {}

struct SpeechAPIController: RouteCollection {
    private let makeSpeechService: () -> any SpeechServing

    init(makeSpeechService: @escaping () -> any SpeechServing = SpeechAPIController.defaultMakeSpeechService) {
        self.makeSpeechService = makeSpeechService
    }

    /// Default factory: route HTTP transcription through the macOS 26
    /// `SpeechAnalyzer` pipeline via `PipelineSpeechService`. The pipeline's
    /// `AnalysisContext.contextualStrings` honors the bundled vocab list more
    /// strongly than `SFSpeechURLRecognitionRequest.contextualStrings` did
    /// on the legacy path — that's the change that's expected to close the
    /// Gate B (technical-English) WER gap against whisper-cpp.
    ///
    /// A single `PipelineSpeechService` instance is shared across requests
    /// so the underlying transcriber pool stays warm between calls. The
    /// service constructs its pipeline lazily on first use.
    private static let sharedPipelineService: any SpeechServing = {
        if #available(macOS 13.0, *) {
            return PipelineSpeechService()
        }
        return SpeechService()
    }()

    private static func defaultMakeSpeechService() -> any SpeechServing {
        return sharedPipelineService
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        let speech = v1.grouped("audio")
        speech.on(.POST, "transcriptions", body: .collect(maxSize: "50mb"), use: transcribe)
        speech.on(.OPTIONS, "transcriptions", use: handleOptions)
    }

    private func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    private func transcribe(req: Request) async throws -> Response {
        guard #available(macOS 13.0, *) else {
            throw Abort(.serviceUnavailable, reason: "Speech recognition requires macOS 13.0 or later")
        }

        struct TranscriptionRequest: Content {
            let file: String?
            let data: String?
            let format: String?
            // OpenAI-compatible field name is `language`; accept `locale` as
            // an alias for callers that already speak our older shape.
            let language: String?
            let locale: String?
            // OpenAI-compatible per-request vocabulary hint. Merged with the
            // bundled / env / project contextual vocab by the resolver.
            let prompt: String?
            // OpenAI-compatible response format selector: text, json,
            // verbose_json. srt/vtt deferred — those need timing-formatted
            // text emission separate from the JSON shape.
            let response_format: String?
            // OpenAI-compatible timestamp granularity selector. Array of
            // "word" / "segment". Only meaningful when response_format is
            // verbose_json.
            let timestamp_granularities: [String]?
        }

        let body = try req.content.decode(TranscriptionRequest.self)
        let locale = body.language ?? body.locale ?? "en-US"
        let options = SpeechRequestOptions(locale: locale, prompt: body.prompt)
        let format = (body.response_format ?? "json").lowercased()
        let wantWordTimings = body.timestamp_granularities?.contains("word") ?? false
        let service = makeSpeechService()
        var cleanupURLs: [URL] = []

        defer {
            for url in cleanupURLs {
                try? FileManager.default.removeItem(at: url)
            }
        }

        do {
            let filePath: String
            if let file = body.file, !file.isEmpty {
                filePath = try Self.sanitizeAudioPath(file)
            } else if let data = body.data, !data.isEmpty {
                let ext = try Self.validatedExtension(body.format ?? "wav")
                let tempURL = try Self.writeTempAudio(base64: data, ext: ext)
                cleanupURLs.append(tempURL)
                filePath = tempURL.path
            } else {
                throw Abort(.badRequest, reason: "Either 'file' path or 'data' (base64) is required")
            }

            switch format {
            case "verbose_json":
                guard let rich = service as? RichSpeechServing else {
                    // Fallback path: the active backend doesn't expose
                    // word/segment timings (legacy SFSpeechRecognizer).
                    // Emit a verbose_json shape with text/duration/language
                    // populated and the timing arrays nil so callers know
                    // the data isn't available rather than silently
                    // pretending an empty list = no words.
                    let text = try await service.transcribe(from: filePath, options: options)
                    let response = VerboseTranscriptionResponse(
                        task: "transcribe",
                        language: locale,
                        duration: 0,
                        text: text,
                        segments: nil,
                        words: nil,
                        languageReassessed: false
                    )
                    return try Self.jsonResponse(response)
                }
                let result = try await rich.transcribeRich(
                    from: filePath,
                    options: options,
                    wantWordTimings: wantWordTimings
                )
                let response = VerboseTranscriptionResponse(
                    task: "transcribe",
                    language: result.language,
                    duration: Double(result.durationMs) / 1000.0,
                    text: result.text,
                    segments: result.segments?.enumerated().map { idx, seg in
                        VerboseSegment(
                            id: idx,
                            start: Double(seg.startMs) / 1000.0,
                            end: Double(seg.endMs) / 1000.0,
                            text: seg.text,
                            avgConfidence: seg.meanConfidence
                        )
                    },
                    words: wantWordTimings
                        ? result.words?.map { w in
                            VerboseWord(
                                word: w.word,
                                start: Double(w.startMs) / 1000.0,
                                end: Double(w.endMs) / 1000.0,
                                confidence: w.confidence
                            )
                        }
                        : nil,
                    languageReassessed: result.languageReassessed
                )
                return try Self.jsonResponse(response)

            case "text":
                let text = try await service.transcribe(from: filePath, options: options)
                let httpResponse = Response(status: .ok)
                httpResponse.headers.add(name: .contentType, value: "text/plain; charset=utf-8")
                httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
                httpResponse.body = .init(string: text)
                return httpResponse

            default: // "json" — backwards-compatible default
                let text = try await service.transcribe(from: filePath, options: options)
                let response = SpeechTranscriptionResponse(
                    object: "speech.transcription",
                    text: text,
                    locale: locale
                )
                return try Self.jsonResponse(response)
            }
        } catch let speechError as SpeechError {
            throw Abort(Self.httpStatus(for: speechError), reason: speechError.localizedDescription)
        }
    }

    private static func jsonResponse<T: Content>(_ payload: T) throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(payload)
        return httpResponse
    }

    static func httpStatus(for error: SpeechError) -> HTTPStatus {
        switch error {
        case .fileNotFound:
            return .notFound
        case .unsupportedFormat:
            return .badRequest
        case .requestTooLarge:
            return .payloadTooLarge
        case .authorizationDenied:
            return .forbidden
        case .onDeviceNotAvailable, .platformUnavailable:
            return .serviceUnavailable
        case .noSpeechFound, .recognitionFailed:
            return .unprocessableEntity
        }
    }

    // MARK: - Chat completions integration

    /// Run Apple Speech transcription on every `input_audio` part in the request messages
    /// and return the transcribed text in order, along with any temp files that
    /// need to be cleaned up by the caller.
    ///
    /// Mirrors `VisionAPIController.extractOCRTextFromMessages`: the sole caller
    /// (the audio-only bypass in `ChatCompletionsController`) streams the
    /// transcriptions directly instead of routing them back through a Foundation
    /// Model prompt, so message reconstruction would be dead code.
    static func extractTranscriptionFromMessages(
        _ messages: [Message],
        options: SpeechRequestOptions,
        service injectedService: (any SpeechServing)? = nil
    ) async throws -> (transcriptionTexts: [String], cleanupURLs: [URL]) {
        guard #available(macOS 13.0, *) else {
            return ([], [])
        }

        let service: any SpeechServing = injectedService ?? SpeechService()
        var transcriptionTexts: [String] = []
        var cleanupURLs: [URL] = []
        var audioIndex = 0

        do {
            for message in messages {
                guard let content = message.content, case .parts(let parts) = content else {
                    continue
                }

                for part in parts where part.type == "input_audio" {
                    guard let inputAudio = part.input_audio else { continue }
                    let ext = try validatedExtension(inputAudio.format.isEmpty ? "wav" : inputAudio.format)
                    let tempURL = try writeTempAudio(base64: inputAudio.data, ext: ext)
                    cleanupURLs.append(tempURL)
                    let transcription = try await service.transcribe(from: tempURL.path, options: options)
                    audioIndex += 1
                    let labeled = "[Apple Speech transcription \(audioIndex)]\n\(transcription)"
                    transcriptionTexts.append(labeled)
                }
            }
        } catch {
            // On partial failure the caller never receives `cleanupURLs`, so any temp files
            // written before the throw would leak.  Clean up here and rethrow.
            for url in cleanupURLs {
                try? FileManager.default.removeItem(at: url)
            }
            throw error
        }

        return (transcriptionTexts, cleanupURLs)
    }

    // MARK: - Helpers

    /// Validate and resolve a file path for the API endpoint.
    /// Resolves symlinks, rejects directories, and enforces audio extension allowlist.
    private static func sanitizeAudioPath(_ raw: String) throws -> String {
        let expanded = NSString(string: raw).expandingTildeInPath
        let resolved = URL(fileURLWithPath: expanded).resolvingSymlinksInPath().path
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: resolved, isDirectory: &isDir) else {
            throw SpeechError.fileNotFound
        }
        guard !isDir.boolValue else {
            throw SpeechError.unsupportedFormat
        }
        let ext = URL(fileURLWithPath: resolved).pathExtension.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(ext) else {
            throw SpeechError.unsupportedFormat
        }
        return resolved
    }

    /// Validate that ext is a supported audio extension before using it in a filename.
    private static func validatedExtension(_ ext: String) throws -> String {
        let clean = ext.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(clean) else {
            throw SpeechError.unsupportedFormat
        }
        return clean
    }

    private static func writeTempAudio(base64: String, ext: String) throws -> URL {
        guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else {
            throw Abort(.badRequest, reason: "Invalid base64 audio data")
        }
        if data.count > SpeechRequestOptions.defaultMaxFileBytes {
            throw SpeechError.requestTooLarge(actualBytes: data.count, maxBytes: SpeechRequestOptions.defaultMaxFileBytes)
        }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_speech_\(UUID().uuidString).\(ext)")
        try data.write(to: tempURL)
        return tempURL
    }
}
