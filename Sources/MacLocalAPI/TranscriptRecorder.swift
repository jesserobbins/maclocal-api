import Foundation
import CryptoKit

/// Backend-neutral assistant turn captured for transcript recording. Both the
/// Foundation and MLX controllers assemble one of these from their own response
/// types so the dedup/format/IO logic lives in exactly one place.
struct RecordedAssistant {
    let content: String?
    let reasoning: String?
    let toolCalls: [ResponseToolCall]?
    let finishReason: String
    let promptTokens: Int?
    let completionTokens: Int?

    init(
        content: String?,
        reasoning: String? = nil,
        toolCalls: [ResponseToolCall]? = nil,
        finishReason: String,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil
    ) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

/// Records chat sessions as OpenAI-shaped JSONL — one `<sessionId>.jsonl` per
/// session, a `session_meta` first line followed by one line per message. The
/// format is the interface consumed by agentsview's Hermes-clone parser; keep
/// it stable.
///
/// AFM re-receives the entire conversation history on every call, so a naive
/// append would duplicate every prior turn. The recorder tracks a persisted
/// message count per session and appends only the genuinely new turn(s) plus
/// the new assistant line. If a client truncates/edits history (sends fewer
/// messages than already persisted), the session is rerouted to a new
/// suffixed file rather than corrupting the existing one.
///
/// Per-session serialization is provided by actor isolation: the count map and
/// file appends are a single critical section, so lines never interleave.
actor TranscriptRecorder {
    private let transcriptDir: URL

    /// Maps a session id to the number of messages already persisted to its
    /// file (request messages + each assistant response line). The invariant
    /// after a successful record() is: count == requestMessages.count + 1.
    private var persistedCount: [String: Int] = [:]

    /// Maps a session id to the on-disk file currently receiving its lines.
    /// Diverges from `<sessionId>.jsonl` only after a truncated-history
    /// fallback reroutes the session to a suffixed file.
    private var activeFile: [String: URL] = [:]

    /// Naive (zone-less) microsecond ISO-8601, e.g. 2026-06-17T15:36:20.123456.
    /// agentsview interprets a zone-less timestamp as local wall-clock.
    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        return f
    }()

    init(transcriptDir: URL) {
        self.transcriptDir = transcriptDir
    }

    /// Record one completed turn: the new request messages since the last call
    /// plus the assistant response. Best-effort — IO failures are logged to
    /// stderr and never propagate to the request path.
    func record(
        sessionId: String,
        model: String,
        requestMessages: [Message],
        assistant: RecordedAssistant
    ) {
        let sanitized = Self.sanitize(sessionId)
        let count = persistedCount[sanitized] ?? 0

        // Truncated/edited history: the client sent fewer messages than we have
        // already persisted. Reroute to a fresh suffixed file rather than
        // corrupting the existing transcript.
        if !requestMessages.isEmpty && requestMessages.count < count {
            let fresh = Self.suffixedID(sanitized, existing: Set(persistedCount.keys))
            persistedCount[fresh] = 0
            activeFile[fresh] = fileURL(for: fresh)
            record(sessionId: fresh, model: model, requestMessages: requestMessages, assistant: assistant)
            return
        }

        let file = activeFile[sanitized] ?? fileURL(for: sanitized)
        activeFile[sanitized] = file

        var lines: [String] = []

        if count == 0 {
            lines.append(metaLine(sessionId: sanitized, model: model))
            // First call: persist every request message.
            for message in requestMessages {
                lines.append(messageLine(message))
            }
        } else {
            // Subsequent call: persist only the genuinely new request messages.
            // count includes the previous assistant line, which the client
            // echoes back as one message, so messages[count-1...] would re-add
            // it. The new messages are everything from index `count` onward,
            // because count == previousRequest.count + 1 (the assistant line).
            if requestMessages.count > count {
                for message in requestMessages[count...] {
                    lines.append(messageLine(message))
                }
            }
        }

        lines.append(assistantLine(assistant))

        append(lines, to: file)

        // New persisted count = all request messages + the assistant line.
        persistedCount[sanitized] = requestMessages.count + 1
    }

    // MARK: - Line builders

    private func metaLine(sessionId: String, model: String) -> String {
        encode([
            "role": "session_meta",
            "session_id": sessionId,
            "model": model,
            "platform": "afm",
            "timestamp": timestamp(),
        ])
    }

    private func messageLine(_ message: Message) -> String {
        var obj: [String: Any] = [
            "role": message.role,
            "content": message.textContent,
            "timestamp": timestamp(),
        ]
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            obj["tool_calls"] = toolCalls.map { tc -> [String: Any] in
                [
                    "id": tc.id,
                    "type": "function",
                    "function": ["name": tc.function.name, "arguments": tc.function.arguments],
                ]
            }
        }
        if let toolCallId = message.toolCallId {
            obj["tool_call_id"] = toolCallId
        }
        if let name = message.name {
            obj["name"] = name
        }
        return encode(obj)
    }

    private func assistantLine(_ assistant: RecordedAssistant) -> String {
        var obj: [String: Any] = [
            "role": "assistant",
            "content": assistant.content ?? "",
            "finish_reason": assistant.finishReason,
            "timestamp": timestamp(),
        ]
        if let reasoning = assistant.reasoning, !reasoning.isEmpty {
            obj["reasoning"] = reasoning
        }
        if let toolCalls = assistant.toolCalls, !toolCalls.isEmpty {
            obj["tool_calls"] = toolCalls.map { tc -> [String: Any] in
                [
                    "id": tc.id,
                    "type": "function",
                    "function": ["name": tc.function.name, "arguments": tc.function.arguments],
                ]
            }
        }
        if let prompt = assistant.promptTokens, let completion = assistant.completionTokens {
            obj["usage"] = ["prompt_tokens": prompt, "completion_tokens": completion]
        }
        return encode(obj)
    }

    // MARK: - IO

    private func fileURL(for sessionId: String) -> URL {
        transcriptDir.appendingPathComponent("\(sessionId).jsonl")
    }

    private func append(_ lines: [String], to file: URL) {
        guard !lines.isEmpty else { return }
        let payload = Data((lines.joined(separator: "\n") + "\n").utf8)
        do {
            try FileManager.default.createDirectory(at: transcriptDir, withIntermediateDirectories: true)
            if let handle = try? FileHandle(forWritingTo: file) {
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: payload)
            } else {
                try payload.write(to: file, options: .atomic)
            }
        } catch {
            FileHandle.standardError.write(Data("[transcript] write failed for \(file.lastPathComponent): \(error)\n".utf8))
        }
    }

    private func encode(_ obj: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    private func timestamp() -> String {
        Self.timestampFormatter.string(from: Date())
    }

    // MARK: - Session id

    /// Restrict to [A-Za-z0-9._-]; hash if the result would be unwieldy or empty.
    static func sanitize(_ id: String) -> String {
        let allowed = Set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
        let cleaned = String(id.map { allowed.contains($0) ? $0 : "_" })
        if cleaned.isEmpty || cleaned.count > 128 {
            return hashed(id)
        }
        return cleaned
    }

    private static func hashed(_ value: String) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        return digest.map { String(format: "%02x", $0) }.joined().prefix(32).description
    }

    private static func suffixedID(_ base: String, existing: Set<String>) -> String {
        var n = 2
        while existing.contains("\(base)-\(n)") { n += 1 }
        return "\(base)-\(n)"
    }

    /// Stable per-process nonce mixed into synthesized session ids so a server
    /// restart never reopens a prior file, while repeated calls within one
    /// process whose first user message matches resolve to the same session.
    private static let processNonce = UUID().uuidString

    /// Resolve a session id in priority order: an explicit `X-Session-Id`
    /// header (preferred — clients should set it), then the OpenAI `user` body
    /// field, then a content-stable synthesized id derived from the first user
    /// message and the process nonce. Returned ids are already sanitized.
    static func resolveSessionID(header: String?, bodyUser: String?, firstUserMessage: String?) -> String {
        if let header, !header.trimmingCharacters(in: .whitespaces).isEmpty {
            return sanitize(header)
        }
        if let bodyUser, !bodyUser.trimmingCharacters(in: .whitespaces).isEmpty {
            return sanitize(bodyUser)
        }
        return hashed("\(processNonce)|\(firstUserMessage ?? "")")
    }
}
