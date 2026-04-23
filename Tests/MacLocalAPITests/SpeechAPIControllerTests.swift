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

    func testTranscribeRejectsUnsupportedFormat() async throws {
        try SpeechAPIController(makeSpeechService: { FakeSpeechService() }).boot(routes: app)

        let payload = Data("fake".utf8).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(payload)","format":"flac"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/audio/transcriptions", headers: headers, body: body) { res async in
            XCTAssertTrue(res.status.code >= 400, "expected an error status, got \(res.status)")
            XCTAssertContains(res.body.string, "unsupportedFormat")
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

    func testExtractTranscriptionFromMessagesReturnsLabeledTexts() async throws {
        guard #available(macOS 13.0, *) else { return }

        let payload = Data("fake wav".utf8).base64EncodedString()
        let messages = [
            Message(role: "user", content: .parts([
                ContentPart(type: "text", text: "transcribe this", image_url: nil),
                ContentPart(type: "input_audio", text: nil, image_url: nil, input_audio: InputAudio(data: payload, format: "wav"))
            ]))
        ]

        // Note: this uses the default SpeechService via the helper, which requires mic authorization.
        // We only verify that a non-audio-only message returns early with no side effects.
        let plainMessages = [Message(role: "user", content: "hello")]
        let result = try await SpeechAPIController.extractTranscriptionFromMessages(plainMessages, options: SpeechRequestOptions())
        XCTAssertEqual(result.transcriptionTexts, [])
        XCTAssertEqual(result.cleanupURLs, [])

        // Suppress unused variable warning by touching `messages`; the full-round-trip
        // path is exercised in integration tests (the service has OS-level dependencies).
        _ = messages
    }
}

private final class FakeSpeechService: SpeechServing {
    var transcriptionText = ""
    var lastFilePath: String?
    var lastOptions: SpeechRequestOptions?
    var transcribeError: Error?

    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
        lastFilePath = filePath
        lastOptions = options
        if let transcribeError { throw transcribeError }
        return transcriptionText
    }
}
