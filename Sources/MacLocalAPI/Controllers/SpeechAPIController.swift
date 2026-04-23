import Vapor
import Foundation

struct SpeechTranscriptionResponse: Content {
    let object: String
    let text: String
    let locale: String
}

protocol SpeechServing {
    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String
}

extension SpeechService: SpeechServing {}

struct SpeechAPIController: RouteCollection {
    private let makeSpeechService: () -> any SpeechServing

    init(makeSpeechService: @escaping () -> any SpeechServing = { SpeechService() }) {
        self.makeSpeechService = makeSpeechService
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
            let locale: String?
        }

        let body = try req.content.decode(TranscriptionRequest.self)
        let locale = body.locale ?? "en-US"
        let options = SpeechRequestOptions(locale: locale)
        let service = makeSpeechService()
        var cleanupURLs: [URL] = []

        defer {
            for url in cleanupURLs {
                try? FileManager.default.removeItem(at: url)
            }
        }

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

        let text = try await service.transcribe(from: filePath, options: options)

        let response = SpeechTranscriptionResponse(
            object: "speech.transcription",
            text: text,
            locale: locale
        )
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(response)
        return httpResponse
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
