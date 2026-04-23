import Foundation

/// Merges contextual-vocabulary hints from the four precedence sources and
/// produces the final `[String]` handed to `AnalysisContext.contextualStrings`
/// under the `.general` tag.
///
/// Precedence (high to low) — strings from higher-precedence sources appear
/// first after merge/dedup:
///
/// 1. Per-request `prompt` field (passed to `resolve(prompt:)`).
/// 2. `MACAFM_SPEECH_VOCAB_FILE` environment variable (path to a plaintext file).
/// 3. Project vocab at `<server_cwd>/.afm/speech-vocab.txt`.
/// 4. Bundled default at `Resources/speech-vocab/en.txt` (always on).
///
/// The bundled/env/project sources are loaded once at construction time and
/// held in memory; per-request resolution is a cheap in-memory union.
public final class ContextualVocabResolver: Sendable {
    /// Maximum entries emitted per resolve call. Apple's `contextualStrings`
    /// dictionary has no documented hard cap; we pick a pragmatic upper bound
    /// that keeps payloads small enough for the analyzer to consume in a
    /// single setContext call without measurable overhead.
    public static let maxResolvedEntries: Int = 4096

    /// Maximum tokens we extract from a per-request `prompt` field. OpenAI's
    /// `prompt` is a free-text hint; we split on whitespace and keep the
    /// first N non-trivial tokens.
    public static let maxPromptTokens: Int = 256

    private let bundledEntries: [String]
    private let envEntries: [String]
    private let projectEntries: [String]

    public init(
        bundle: Bundle = .module,
        envFilePath: String? = ProcessInfo.processInfo.environment["MACAFM_SPEECH_VOCAB_FILE"],
        projectFilePath: String? = ContextualVocabResolver.defaultProjectFilePath()
    ) throws {
        self.bundledEntries = try ContextualVocabResolver.loadBundled(bundle: bundle)
        self.envEntries = ContextualVocabResolver.loadFileIfExists(path: envFilePath)
        self.projectEntries = ContextualVocabResolver.loadFileIfExists(path: projectFilePath)
    }

    /// Resolve the merged contextual-strings list for a single request.
    public func resolve(prompt: String?, locale: String) -> [String] {
        let promptTokens = ContextualVocabResolver.tokenize(prompt: prompt)

        // Merge high-to-low, dedup on a case-folded key but keep the first
        // occurrence's original casing (which will be the per-request prompt
        // if supplied).
        var seen: Set<String> = []
        var result: [String] = []
        result.reserveCapacity(promptTokens.count + envEntries.count + projectEntries.count + bundledEntries.count)

        for source in [promptTokens, envEntries, projectEntries, bundledEntries] {
            for entry in source {
                let key = entry.lowercased()
                if seen.insert(key).inserted {
                    result.append(entry)
                    if result.count >= ContextualVocabResolver.maxResolvedEntries {
                        return result
                    }
                }
            }
        }
        return result
    }

    // MARK: - Source loading

    private static func loadBundled(bundle: Bundle) throws -> [String] {
        guard let url = bundle.url(
            forResource: "en",
            withExtension: "txt",
            subdirectory: "speech-vocab"
        ) else {
            // During development the bundle may not include the vocab if the
            // build script hasn't been run yet. Prefer a loud explicit error
            // over silently shipping zero-config with no bundled vocab —
            // Gate B depends on this file existing.
            throw VocabResolverError.bundledVocabMissing
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        return parse(text: text)
    }

    private static func loadFileIfExists(path: String?) -> [String] {
        guard let path, !path.isEmpty else { return [] }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return []
        }
        return parse(text: text)
    }

    private static func defaultProjectFilePath() -> String? {
        // Conventional project-level location, resolved relative to the
        // server's working directory — matching how other AFM config reads
        // (.afm/... style paths) work in this repo.
        let cwd = FileManager.default.currentDirectoryPath
        return (cwd as NSString).appendingPathComponent(".afm/speech-vocab.txt")
    }

    /// Parse a plaintext vocab file: one entry per line, `#` comments, blanks
    /// ignored. Duplicates within the file are preserved; dedup happens at
    /// merge time so we don't lose positional hints.
    static func parse(text: String) -> [String] {
        var out: [String] = []
        text.enumerateLines { line, _ in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { return }
            if trimmed.hasPrefix("#") { return }
            out.append(trimmed)
        }
        return out
    }

    /// Tokenize a per-request `prompt` field. We split conservatively on
    /// whitespace and drop entries that are too short to be meaningful hints
    /// (single-character tokens rarely help contextual bias).
    static func tokenize(prompt: String?) -> [String] {
        guard let prompt, !prompt.isEmpty else { return [] }
        let whole = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var tokens: [String] = []
        if !whole.isEmpty {
            // Always include the full prompt as one bias string — short
            // phrases like "Kubernetes microservices" work better as a
            // single entry than as individual tokens.
            tokens.append(whole)
        }
        let parts = prompt.split(whereSeparator: { $0.isWhitespace })
        for part in parts {
            let t = String(part).trimmingCharacters(in: .punctuationCharacters)
            if t.count >= 2 {
                tokens.append(t)
            }
            if tokens.count >= ContextualVocabResolver.maxPromptTokens {
                break
            }
        }
        return tokens
    }
}

public enum VocabResolverError: Error, LocalizedError {
    case bundledVocabMissing

    public var errorDescription: String? {
        switch self {
        case .bundledVocabMissing:
            return "Bundled speech vocabulary (Resources/speech-vocab/en.txt) not found in module bundle. Run Scripts/build-from-scratch.sh to stage it."
        }
    }
}
