import XCTest
import Foundation
@testable import MacLocalAPI

// dimensions: transcript-recording
//
// Verifies the OpenAI-shaped JSONL transcript format consumed by agentsview's
// Hermes-clone parser: one meta line + one line per message, with history
// dedup so AFM's full-history-per-call protocol does not duplicate turns.

final class TranscriptRecorderTests: XCTestCase {

    private func makeTempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-transcript-tests-\(UUID().uuidString)")
        return dir
    }

    private func readLines(_ url: URL) throws -> [[String: Any]] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return text.split(separator: "\n", omittingEmptySubsequences: true).map { line in
            let data = Data(line.utf8)
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        }
    }

    private func sessionFiles(in dir: URL) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.filter { $0.pathExtension == "jsonl" }.sorted { $0.path < $1.path }
    }

    // MARK: - First-call creation

    func testFirstCallWritesMetaThenAllRequestMessagesThenAssistant() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        let request = [
            Message(role: "system", content: "You are helpful"),
            Message(role: "user", content: "Hello"),
        ]
        await recorder.record(
            sessionId: "sess-1",
            model: "test-model",
            requestMessages: request,
            assistant: RecordedAssistant(content: "Hi there", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 1, "exactly one transcript file")
        let lines = try readLines(files[0])
        // meta, system, user, assistant
        XCTAssertEqual(lines.count, 4)
        XCTAssertEqual(lines[0]["role"] as? String, "session_meta")
        XCTAssertEqual(lines[0]["session_id"] as? String, "sess-1")
        XCTAssertEqual(lines[0]["model"] as? String, "test-model")
        XCTAssertEqual(lines[0]["platform"] as? String, "afm")
        XCTAssertNotNil(lines[0]["timestamp"] as? String)
        XCTAssertEqual(lines[1]["role"] as? String, "system")
        XCTAssertEqual(lines[2]["role"] as? String, "user")
        XCTAssertEqual(lines[2]["content"] as? String, "Hello")
        XCTAssertEqual(lines[3]["role"] as? String, "assistant")
        XCTAssertEqual(lines[3]["content"] as? String, "Hi there")
        XCTAssertEqual(lines[3]["finish_reason"] as? String, "stop")
    }

    // MARK: - Delta append / dedup (the off-by-one trap)

    func testSecondCallAppendsOnlyNewTurnNoHistoryDuplication() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // Turn 1: request [system, user1]
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
            ],
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )

        // Turn 2: client echoes full history [system, user1, assistant1, user2]
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
                Message(role: "assistant", content: "a1"),
                Message(role: "user", content: "u2"),
            ],
            assistant: RecordedAssistant(content: "a2", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 1, "still exactly one file after second turn")
        let lines = try readLines(files[0])

        // meta, sys, u1, a1, u2, a2  — each exactly once
        XCTAssertEqual(lines.count, 6)
        let contents = lines.compactMap { $0["content"] as? String }
        XCTAssertEqual(contents.filter { $0 == "a1" }.count, 1, "a1 must appear exactly once (off-by-one guard)")
        XCTAssertEqual(contents.filter { $0 == "u1" }.count, 1)
        XCTAssertEqual(contents.filter { $0 == "sys" }.count, 1)
        XCTAssertEqual(lines[3]["content"] as? String, "a1")
        XCTAssertEqual(lines[4]["content"] as? String, "u2")
        XCTAssertEqual(lines[5]["content"] as? String, "a2")
    }

    // MARK: - Truncated-history fallback

    func testTruncatedHistoryStartsNewFileRatherThanCorrupting() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // Turn 1: request [system, user1] (count becomes 3)
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
            ],
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )

        // Turn 2: client truncated/edited history — fewer messages than persisted.
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "user", content: "fresh"),
            ],
            assistant: RecordedAssistant(content: "a2", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 2, "truncated history must spawn a new file, not corrupt the first")

        // Original file is untouched: meta, sys, u1, a1
        let original = try readLines(files.first { $0.lastPathComponent == "sess-1.jsonl" }!)
        XCTAssertEqual(original.count, 4)
        XCTAssertEqual(original.last?["content"] as? String, "a1")

        // New file holds the fresh turn: meta, fresh, a2
        let fresh = try readLines(files.first { $0.lastPathComponent != "sess-1.jsonl" }!)
        XCTAssertEqual(fresh[0]["role"] as? String, "session_meta")
        XCTAssertEqual(fresh.last?["content"] as? String, "a2")
    }

    func testSameLengthEditedHistoryStartsNewFileRatherThanCorrupting() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // Turn 1: [system, user1] → persisted prefix is [system, user1, a1].
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
            ],
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )

        // Turn 2: client edits user1 and resends the SAME number of messages
        // (edit-and-resend / regenerate-with-edit). Length is unchanged, so a
        // length-only guard would silently append; content divergence must
        // reroute instead.
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1-edited"),
            ],
            assistant: RecordedAssistant(content: "a2", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 2, "same-length edited history must spawn a new file, not corrupt the first")

        // Original untouched: meta, sys, u1, a1.
        let original = try readLines(files.first { $0.lastPathComponent == "sess-1.jsonl" }!)
        XCTAssertEqual(original.count, 4)
        XCTAssertEqual(original[2]["content"] as? String, "u1")
        XCTAssertEqual(original.last?["content"] as? String, "a1")

        // New file holds the edited turn: meta, sys, u1-edited, a2.
        let edited = try readLines(files.first { $0.lastPathComponent != "sess-1.jsonl" }!)
        XCTAssertEqual(edited[0]["role"] as? String, "session_meta")
        let editedContents = edited.compactMap { $0["content"] as? String }
        XCTAssertTrue(editedContents.contains("u1-edited"))
        XCTAssertEqual(edited.last?["content"] as? String, "a2")
    }

    func testEditedEarlierTurnWithGrownHistoryStartsNewFile() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // Turn 1: [system, user1].
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
            ],
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )

        // Turn 2: client edits the FIRST user turn but also adds a new turn, so
        // the history is longer than persisted. count > persisted.count is true,
        // but the persisted prefix no longer matches → must reroute, not append.
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1-edited"),
                Message(role: "assistant", content: "a1"),
                Message(role: "user", content: "u2"),
            ],
            assistant: RecordedAssistant(content: "a2", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 2, "edited earlier turn (even with grown history) must reroute")
        let original = try readLines(files.first { $0.lastPathComponent == "sess-1.jsonl" }!)
        XCTAssertEqual(original.count, 4, "original transcript must be untouched")
    }

    func testEchoedAssistantWithMatchingPrefixAppendsToSameFile() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // Turn 1: request [sys, u1] → persisted prefix [sys, u1, a1].
        await recorder.record(
            sessionId: "sess-1", model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
            ],
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )
        // Turn 2: client echoes the prior assistant back verbatim and asks for
        // another reply with no new user turn. The persisted prefix matches
        // exactly, so the new assistant is appended to the same file rather than
        // rerouting. (The echoed assistant message must fingerprint identically
        // to the persisted assistant line for this to hold.)
        await recorder.record(
            sessionId: "sess-1", model: "m",
            requestMessages: [
                Message(role: "system", content: "sys"),
                Message(role: "user", content: "u1"),
                Message(role: "assistant", content: "a1"),
            ],
            assistant: RecordedAssistant(content: "a2", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 1, "matching prefix stays in one file")
        let lines = try readLines(files[0])
        // meta, sys, u1, a1, a2 — a1 appears exactly once.
        XCTAssertEqual(lines.count, 5)
        XCTAssertEqual(lines.compactMap { $0["content"] as? String }.filter { $0 == "a1" }.count, 1)
        XCTAssertEqual(lines[4]["role"] as? String, "assistant")
        XCTAssertEqual(lines[4]["content"] as? String, "a2")
    }

    func testNaturalRegenerateReroutesBecausePriorAssistantNotEchoed() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        let request = [
            Message(role: "system", content: "sys"),
            Message(role: "user", content: "u1"),
        ]
        // Turn 1 → persisted prefix [sys, u1, a1].
        await recorder.record(
            sessionId: "sess-1", model: "m",
            requestMessages: request,
            assistant: RecordedAssistant(content: "a1", finishReason: "stop")
        )
        // Natural regenerate: client resends the original request (WITHOUT the
        // prior assistant) for a fresh reply. Persisted [sys, u1, a1] is not a
        // prefix of the shorter [sys, u1], so this reroutes rather than
        // corrupting the first transcript. One file per regenerate is the
        // deliberate, conservative choice — see the refine ledger.
        await recorder.record(
            sessionId: "sess-1", model: "m",
            requestMessages: request,
            assistant: RecordedAssistant(content: "a1-regenerated", finishReason: "stop")
        )

        let files = sessionFiles(in: dir)
        XCTAssertEqual(files.count, 2, "natural regenerate reroutes to a fresh file")
        let original = try readLines(files.first { $0.lastPathComponent == "sess-1.jsonl" }!)
        XCTAssertEqual(original.count, 4)
        XCTAssertEqual(original.last?["content"] as? String, "a1")
        let regen = try readLines(files.first { $0.lastPathComponent != "sess-1.jsonl" }!)
        XCTAssertEqual(regen.last?["content"] as? String, "a1-regenerated")
    }

    func testRequestToolMessagesRoundTripThroughRecord() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        // A multi-turn request carrying an assistant-with-tool_calls message and
        // a tool result message, as a client would echo back after a tool call.
        let json = """
        [
          {"role": "user", "content": "weather in SF?"},
          {"role": "assistant", "content": null,
           "tool_calls": [{"id": "call_1", "type": "function",
                           "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"SF\\"}"}}]},
          {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "62F"}
        ]
        """
        let messages = try JSONDecoder().decode([Message].self, from: Data(json.utf8))

        await recorder.record(
            sessionId: "sess-1", model: "m",
            requestMessages: messages,
            assistant: RecordedAssistant(content: "It's 62F in SF.", finishReason: "stop")
        )

        let lines = try readLines(sessionFiles(in: dir)[0])
        // meta, user, assistant(tool_calls), tool, assistant.
        XCTAssertEqual(lines.count, 5)

        let assistantToolCall = lines[2]
        XCTAssertEqual(assistantToolCall["role"] as? String, "assistant")
        let tcs = assistantToolCall["tool_calls"] as? [[String: Any]]
        XCTAssertEqual(tcs?.count, 1)
        XCTAssertEqual(tcs?[0]["id"] as? String, "call_1")
        XCTAssertEqual((tcs?[0]["function"] as? [String: Any])?["name"] as? String, "get_weather")

        let toolResult = lines[3]
        XCTAssertEqual(toolResult["role"] as? String, "tool")
        XCTAssertEqual(toolResult["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(toolResult["name"] as? String, "get_weather")
        XCTAssertEqual(toolResult["content"] as? String, "62F")
    }

    // MARK: - tool_calls shape

    func testAssistantToolCallsSerializeInOpenAIShape() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)

        let tc = ResponseToolCall(
            index: 0,
            id: "call_abc",
            type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: "{\"city\":\"SF\"}")
        )
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [Message(role: "user", content: "weather?")],
            assistant: RecordedAssistant(content: nil, toolCalls: [tc], finishReason: "tool_calls")
        )

        let lines = try readLines(sessionFiles(in: dir)[0])
        let assistant = lines.last!
        XCTAssertEqual(assistant["finish_reason"] as? String, "tool_calls")
        let toolCalls = assistant["tool_calls"] as? [[String: Any]]
        XCTAssertEqual(toolCalls?.count, 1)
        XCTAssertEqual(toolCalls?[0]["id"] as? String, "call_abc")
        let fn = toolCalls?[0]["function"] as? [String: Any]
        XCTAssertEqual(fn?["name"] as? String, "get_weather")
        XCTAssertEqual(fn?["arguments"] as? String, "{\"city\":\"SF\"}")
    }

    // MARK: - Session id resolution

    func testSessionIDPrefersHeaderThenBodyUserThenSynthesized() {
        // Header wins over everything.
        XCTAssertEqual(
            TranscriptRecorder.resolveSessionID(header: "hdr-1", bodyUser: "user-1", firstUserMessage: "hi"),
            "hdr-1"
        )
        // Body `user` is used when no header.
        XCTAssertEqual(
            TranscriptRecorder.resolveSessionID(header: nil, bodyUser: "user-1", firstUserMessage: "hi"),
            "user-1"
        )
        // Blank header falls through to body user.
        XCTAssertEqual(
            TranscriptRecorder.resolveSessionID(header: "   ", bodyUser: "user-1", firstUserMessage: "hi"),
            "user-1"
        )
    }

    func testSynthesizedSessionIDIsStableForSameFirstMessageWithinProcess() {
        // No header, no body user → synthesized from first user message + process
        // nonce. Must be stable across calls so a multi-turn conversation whose
        // replayed first message is identical lands in one file.
        let a = TranscriptRecorder.resolveSessionID(header: nil, bodyUser: nil, firstUserMessage: "Hello there")
        let b = TranscriptRecorder.resolveSessionID(header: nil, bodyUser: nil, firstUserMessage: "Hello there")
        XCTAssertEqual(a, b, "same first message must yield the same synthesized id within a process")

        let c = TranscriptRecorder.resolveSessionID(header: nil, bodyUser: nil, firstUserMessage: "Different opening")
        XCTAssertNotEqual(a, c, "different first messages must not collide")

        // Synthesized ids are filesystem-safe.
        let pattern = #"^[A-Za-z0-9._-]+$"#
        XCTAssertNotNil(a.range(of: pattern, options: .regularExpression))
    }

    func testSanitizeRestrictsToSafeCharacters() {
        XCTAssertEqual(TranscriptRecorder.sanitize("ok.session_id-1"), "ok.session_id-1")
        XCTAssertEqual(TranscriptRecorder.sanitize("a/b c:d"), "a_b_c_d")
        // Empty → hashed (non-empty, safe).
        let hashed = TranscriptRecorder.sanitize("")
        XCTAssertFalse(hashed.isEmpty)
        XCTAssertNotNil(hashed.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression))
    }

    // MARK: - Timestamp format

    func testTimestampIsNaiveMicrosecondISO8601() async throws {
        let dir = makeTempDir()
        let recorder = TranscriptRecorder(transcriptDir: dir)
        await recorder.record(
            sessionId: "sess-1",
            model: "m",
            requestMessages: [Message(role: "user", content: "hi")],
            assistant: RecordedAssistant(content: "yo", finishReason: "stop")
        )
        let lines = try readLines(sessionFiles(in: dir)[0])
        let ts = lines[0]["timestamp"] as? String ?? ""
        // e.g. 2026-06-17T15:36:20.123456 — naive (no zone), 6 fractional digits.
        let pattern = #"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$"#
        XCTAssertNotNil(ts.range(of: pattern, options: .regularExpression),
                        "timestamp '\(ts)' must match naive microsecond ISO8601")
    }
}
