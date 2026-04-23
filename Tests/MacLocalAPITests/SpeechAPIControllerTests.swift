import XCTest
import Vapor
import XCTVapor

@testable import MacLocalAPI

final class SpeechAPIControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testTranscribeWithBase64DataReturnsTextAndLocale() async throws {
        let service = FakeSpeechService()
        service.transcriptionText = "hello world"
        try SpeechAPIController(makeSpeechService: { service }).boot(routes: app)

        let payload = Data("fake wav".utf8).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(payload)","format":"wav","locale":"en-US"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.headers.first(name: .accessControlAllowOrigin), "*")
            XCTAssertContains(res.body.string, #""object":"speech.transcription""#)
            XCTAssertContains(res.body.string, #""text":"hello world""#)
            XCTAssertContains(res.body.string, #""locale":"en-US""#)
            XCTAssertEqual(service.lastOptions?.locale, "en-US")
        }
    }

    func testTranscribeDefaultsToEnUSWhenLocaleOmitted() async throws {
        let service = FakeSpeechService()
        service.transcriptionText = "hi"
        try SpeechAPIController(makeSpeechService: { service }).boot(routes: app)

        let payload = Data("fake wav".utf8).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(payload)","format":"wav"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(service.lastOptions?.locale, "en-US")
        }
    }

    func testTranscribeRejectsUnsupportedFormatWith400() async throws {
        try SpeechAPIController(makeSpeechService: { FakeSpeechService() }).boot(routes: app)

        let payload = Data("fake".utf8).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(payload)","format":"flac"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "Unsupported audio format")
        }
    }

    func testTranscribeMapsSpeechServiceErrorsToExpectedStatuses() async throws {
        let cases: [(error: SpeechError, expected: HTTPResponseStatus)] = [
            (.fileNotFound, .notFound),
            (.unsupportedFormat, .badRequest),
            (.requestTooLarge(actualBytes: 100, maxBytes: 50), .payloadTooLarge),
            (.authorizationDenied, .forbidden),
            (.onDeviceNotAvailable, .serviceUnavailable),
            (.platformUnavailable, .serviceUnavailable),
            (.noSpeechFound, .unprocessableEntity),
            (.recognitionFailed("boom"), .unprocessableEntity)
        ]

        for (err, expected) in cases {
            // Fresh app per case so each test gets its own route tree.
            let perCaseApp = try await Application.make(.testing)

            let service = FakeSpeechService()
            service.transcribeError = err
            try SpeechAPIController(makeSpeechService: { service }).boot(routes: perCaseApp)

            let payload = Data("fake wav".utf8).base64EncodedString()
            let body = ByteBuffer(string: #"{"data":"\#(payload)","format":"wav"}"#)
            var headers = HTTPHeaders()
            headers.contentType = .json

            try await perCaseApp.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
                XCTAssertEqual(res.status, expected, "\(err) should map to \(expected), got \(res.status)")
            }

            try await perCaseApp.asyncShutdown()
        }
    }

    func testTranscribeRejectsMissingInput() async throws {
        try SpeechAPIController(makeSpeechService: { FakeSpeechService() }).boot(routes: app)

        let body = ByteBuffer(string: "{}")
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "'file' path or 'data'")
        }
    }

    func testHandleOptionsReturnsCORSHeaders() async throws {
        try SpeechAPIController(makeSpeechService: { FakeSpeechService() }).boot(routes: app)

        try await app.testable(method: .running(port: 0)).test(.OPTIONS, "/v1/audio/transcriptions") { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.headers.first(name: .accessControlAllowOrigin), "*")
            XCTAssertContains(res.headers.first(name: .accessControlAllowMethods) ?? "", "POST")
        }
    }

    func testExtractTranscriptionFromMessagesReturnsEmptyWhenNoAudioParts() async throws {
        guard #available(macOS 13.0, *) else { return }

        let plainMessages = [Message(role: "user", content: "hello")]
        let result = try await SpeechAPIController.extractTranscriptionFromMessages(plainMessages, options: SpeechRequestOptions())
        XCTAssertEqual(result.transcriptionTexts, [])
        XCTAssertEqual(result.cleanupURLs, [])
    }

    func testExtractTranscriptionFromMessagesLabelsEachAudioPart() async throws {
        guard #available(macOS 13.0, *) else { return }

        let service = FakeSpeechService()
        service.transcriptionText = "hello"

        let payload = Data("fake wav".utf8).base64EncodedString()
        let messages = [
            Message(role: "user", content: .parts([
                ContentPart(type: "text", text: "transcribe this", image_url: nil),
                ContentPart(type: "input_audio", text: nil, image_url: nil, input_audio: InputAudio(data: payload, format: "wav")),
                ContentPart(type: "input_audio", text: nil, image_url: nil, input_audio: InputAudio(data: payload, format: "wav"))
            ]))
        ]

        let result = try await SpeechAPIController.extractTranscriptionFromMessages(messages, options: SpeechRequestOptions(), service: service)
        XCTAssertEqual(result.transcriptionTexts, [
            "[Apple Speech transcription 1]\nhello",
            "[Apple Speech transcription 2]\nhello"
        ])
        XCTAssertEqual(result.cleanupURLs.count, 2)
        // Cleanup is the caller's responsibility on success; verify the files exist to be cleaned.
        for url in result.cleanupURLs {
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            try? FileManager.default.removeItem(at: url)
        }
    }

    func testExtractTranscriptionFromMessagesCleansUpTempFilesOnThrow() async throws {
        guard #available(macOS 13.0, *) else { return }

        let service = FakeSpeechService()
        service.throwOnCall = 2  // succeed on first, throw on second
        service.transcriptionText = "first"
        service.transcribeError = SpeechError.recognitionFailed("boom")

        let payload = Data("fake wav".utf8).base64EncodedString()
        let messages = [
            Message(role: "user", content: .parts([
                ContentPart(type: "input_audio", text: nil, image_url: nil, input_audio: InputAudio(data: payload, format: "wav")),
                ContentPart(type: "input_audio", text: nil, image_url: nil, input_audio: InputAudio(data: payload, format: "wav"))
            ]))
        ]

        // Capture temp files written so we can verify deletion post-throw.
        let tempDir = FileManager.default.temporaryDirectory
        let beforeSet = Set((try? FileManager.default.contentsOfDirectory(atPath: tempDir.path)) ?? [])

        do {
            _ = try await SpeechAPIController.extractTranscriptionFromMessages(messages, options: SpeechRequestOptions(), service: service)
            XCTFail("expected throw")
        } catch {
            // Expected path: service threw on the second audio part.
        }

        let afterSet = Set((try? FileManager.default.contentsOfDirectory(atPath: tempDir.path)) ?? [])
        let leakedFiles = afterSet.subtracting(beforeSet).filter { $0.hasPrefix("afm_speech_") }
        XCTAssertTrue(leakedFiles.isEmpty, "temp files leaked after throw: \(leakedFiles)")
    }
}

private final class FakeSpeechService: SpeechServing, @unchecked Sendable {
    var transcriptionText = ""
    var lastFilePath: String?
    var lastOptions: SpeechRequestOptions?
    var transcribeError: Error?
    /// If set, throw on the Nth call (1-indexed). Defaults to "always throw if transcribeError is set".
    var throwOnCall: Int?
    private var callCount = 0

    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
        callCount += 1
        lastFilePath = filePath
        lastOptions = options
        if let transcribeError {
            if let throwOn = throwOnCall {
                if callCount == throwOn { throw transcribeError }
            } else {
                throw transcribeError
            }
        }
        return transcriptionText
    }
}
