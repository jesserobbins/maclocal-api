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
/// append would duplicate every prior turn. The recorder fingerprints every
/// message it has persisted per session and appends only the genuinely new
/// turn(s) plus the new assistant line. If a client truncates or edits history
/// — sends fewer messages, or the same/more messages whose earlier content no
/// longer matches what was persisted — the persisted fingerprints are no longer
/// a prefix of the incoming history, so the session is rerouted to a new
/// suffixed file rather than corrupting the existing one. (If the client
/// instead echoes the prior assistant turn back and asks for another reply, the
/// persisted prefix still matches and the new reply is appended as an extra
/// assistant line. A natural regenerate — resending the original request
/// without the prior assistant — is a shorter, non-matching prefix, so it
/// reroutes; one file per regenerate is the conservative choice.)
///
/// Per-session serialization is provided by actor isolation: the fingerprint
/// map and file appends are a single critical section, so lines never interleave.
actor TranscriptRecorder {
    private let transcriptDir: URL

    /// AFM build version and backend name (`foundation` / `mlx`), stamped on the
    /// session_meta line and every assistant line so a transcript identifies
    /// itself as AFM-produced even when a tool reads individual turns. Both are
    /// fixed per server instance (one backend per process on the in-scope paths).
    private let afmVersion: String
    private let backend: String

    /// Maps a session id to the per-message fingerprints already persisted to
    /// its file, in order — request messages plus each assistant response line.
    /// The next call's incoming history must have these as a prefix; otherwise
    /// the client edited/truncated and the session reroutes to a suffixed file.
    private var persistedPrefix: [String: [String]] = [:]

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

    init(transcriptDir: URL, afmVersion: String, backend: String) {
        self.transcriptDir = transcriptDir
        self.afmVersion = afmVersion
        self.backend = backend
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
        var persisted = persistedPrefix[sanitized] ?? []
        let incoming = requestMessages.map { Self.fingerprint($0) }

        // Decide whether we can keep appending to this session's current file, or
        // must redirect it to a fresh suffixed one. Two cases force a redirect:
        //
        //  1. History diverged — the persisted prefix is no longer a prefix of
        //     the incoming history (client truncated or edited an earlier turn).
        //  2. Restart — we have no in-memory prefix for this id (`persisted`
        //     empty) yet a file already exists on disk for it, e.g. a stable
        //     X-Session-Id reused across a server restart. Appending would write
        //     a second session_meta + duplicate history into the existing file.
        //
        // Both redirect by re-pointing `activeFile[sanitized]` at a fresh file
        // and treating this as a first write (persisted = []), so the session's
        // *subsequent* turns adopt the new file and keep appending there rather
        // than fragmenting one file per turn. The original file is untouched.
        let diverged = !persisted.isEmpty && !Self.isPrefix(persisted, of: incoming)
        let restartCollision = persisted.isEmpty
            && activeFile[sanitized] == nil
            && FileManager.default.fileExists(atPath: fileURL(for: sanitized).path)
        if diverged || restartCollision {
            let fresh = Self.suffixedID(sanitized, existing: Set(persistedPrefix.keys), in: transcriptDir)
            let reason = diverged ? "history diverged" : "file exists from a prior run"
            FileHandle.standardError.write(Data("[transcript] session \(sanitized) \(reason); rerouting to \(fresh).jsonl\n".utf8))
            activeFile[sanitized] = fileURL(for: fresh)
            persisted = []
        }

        let file = activeFile[sanitized] ?? fileURL(for: sanitized)
        activeFile[sanitized] = file

        var lines: [String] = []

        if persisted.isEmpty {
            lines.append(metaLine(sessionId: sanitized, model: model))
            // First call: persist every request message.
            for message in requestMessages {
                lines.append(messageLine(message))
            }
        } else {
            // Subsequent call: persist only the request messages beyond the
            // persisted prefix. The persisted prefix already includes the prior
            // assistant line, which the client echoes back, so anything from
            // index `persisted.count` onward is genuinely new.
            if requestMessages.count > persisted.count {
                for message in requestMessages[persisted.count...] {
                    lines.append(messageLine(message))
                }
            }
        }

        lines.append(assistantLine(assistant))

        append(lines, to: file)

        // New persisted prefix = all request fingerprints + the assistant line.
        persistedPrefix[sanitized] = incoming + [Self.assistantFingerprint(assistant)]
    }

    /// Whether `lhs` is a (non-strict) prefix of `rhs`: same elements in order
    /// for the first `lhs.count` positions.
    private static func isPrefix(_ lhs: [String], of rhs: [String]) -> Bool {
        guard lhs.count <= rhs.count else { return false }
        return Array(rhs.prefix(lhs.count)) == lhs
    }

    // MARK: - Line builders

    private func metaLine(sessionId: String, model: String) -> String {
        encode([
            "role": "session_meta",
            "session_id": sessionId,
            "model": model,
            "platform": "afm",
            "afm_version": afmVersion,
            "backend": backend,
            "timestamp": timestamp(),
        ])
    }

    private func messageLine(_ message: Message) -> String {
        // Scope: this recorder captures the text-level API conversation as
        // received. The agentsview JSONL shape types `content` as a string, so a
        // multimodal `.parts` message records its joined text only (image parts
        // are dropped) — text-only is the consumer's native shape, not a lossy
        // workaround. Likewise the recorded messages are the request as sent,
        // not any processed/expanded form the model saw (e.g. vision-OCR).
        // Corollary: an image-only delta (same text, different image) is
        // invisible to prefix matching, since the fingerprint is text-derived.
        //
        // Preserve null content (e.g. a tool-call-only assistant message) as JSON
        // null rather than collapsing it to "", so the consumer can tell a
        // tool-only turn from an empty-text turn. Fingerprinting still uses
        // textContent ("") — these are intentionally independent.
        var obj: [String: Any] = [
            "role": message.role,
            "content": message.content == nil ? NSNull() : message.textContent,
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
            "content": assistant.content == nil ? NSNull() : assistant.content!,
            "finish_reason": assistant.finishReason,
            "platform": "afm",
            "afm_version": afmVersion,
            "backend": backend,
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
        if assistant.promptTokens != nil || assistant.completionTokens != nil {
            var usage: [String: Any] = [:]
            if let prompt = assistant.promptTokens { usage["prompt_tokens"] = prompt }
            if let completion = assistant.completionTokens { usage["completion_tokens"] = completion }
            obj["usage"] = usage
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

    // MARK: - Fingerprints

    /// Content fingerprint of a request message: role, text, and any tool-call
    /// signature. Used for prefix matching so an edited earlier turn is detected
    /// even when the message count is unchanged or larger.
    private static func fingerprint(_ message: Message) -> String {
        var parts = ["\(message.role)\u{1F}\(message.textContent)"]
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            parts.append(toolCalls.map { "\($0.id)\u{1F}\($0.function.name)\u{1F}\($0.function.arguments)" }.joined(separator: "\u{1E}"))
        }
        if let toolCallId = message.toolCallId { parts.append("tcid\u{1F}\(toolCallId)") }
        if let name = message.name { parts.append("name\u{1F}\(name)") }
        return hashed(parts.joined(separator: "\u{1D}"))
    }

    /// Fingerprint of the assistant line as it is persisted, so the next call's
    /// echoed-back assistant message matches the persisted prefix.
    private static func assistantFingerprint(_ assistant: RecordedAssistant) -> String {
        var parts = ["assistant\u{1F}\(assistant.content ?? "")"]
        if let toolCalls = assistant.toolCalls, !toolCalls.isEmpty {
            parts.append(toolCalls.map { "\($0.id)\u{1F}\($0.function.name)\u{1F}\($0.function.arguments)" }.joined(separator: "\u{1E}"))
        }
        return hashed(parts.joined(separator: "\u{1D}"))
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
        return String(digest.map { String(format: "%02x", $0) }.joined().prefix(32))
    }

    /// Pick the lowest `<base>-<n>` (n >= 2) that collides with neither an
    /// in-memory session key nor an existing file on disk. Checking disk too
    /// means a reroute after a server restart (when `existing` is empty) won't
    /// reopen a suffixed file left by a prior process.
    private static func suffixedID(_ base: String, existing: Set<String>, in dir: URL) -> String {
        var n = 2
        while existing.contains("\(base)-\(n)")
            || FileManager.default.fileExists(atPath: dir.appendingPathComponent("\(base)-\(n).jsonl").path) {
            n += 1
        }
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
