import Foundation
import Speech

/// Builds, compiles, and caches an `SFSpeechLanguageModel` containing
/// high-count phrase entries for terms `SpeechTranscriber.contextualStrings`
/// can't reliably recover (Kubernetes, PostgreSQL, ElastiCache, etc.).
///
/// The compiled model is consumed by `DictationTranscriber` via
/// `ContentHint.customizedLanguage(modelConfiguration:)`. Apple does not
/// expose a custom-language-model ContentHint on `SpeechTranscriber` —
/// it's wired specifically to the dictation API path — so this lives
/// alongside the SpeechTranscriber engine rather than inside it.
///
/// Two-stage lifecycle:
///   1. `SFCustomLanguageModelData(locale:, identifier:, version:)` +
///      repeated `insert(phraseCount:)` — purely in-memory training
///      data assembly.
///   2. `data.export(to: assetURL)` writes a training-data archive.
///   3. `SFSpeechLanguageModel.prepareCustomLanguageModel(for:configuration:
///      ignoresCache:completion:)` compiles that archive into the
///      runtime model + vocabulary files referenced by the
///      `Configuration`.
///
/// Both steps cache to ~/Library/Caches/com.macafm/clm/<identifier>/v1/
/// so the second-and-subsequent server starts skip the rebuild
/// (`ignoresCache: false`).
@available(macOS 14.0, *)
public actor CustomLanguageModelManager {

    /// Stable identifier passed to SFSpeechLanguageModel; the system uses
    /// this for cache keying. Bump `modelVersion` when the phrase set
    /// changes so old compiled models are regenerated.
    public static let modelIdentifier = "com.macafm.tech-vocab"
    public static let modelVersion = "1"

    public struct PreparedModel: Sendable {
        public let configuration: SFSpeechLanguageModel.Configuration
    }

    private var cached: PreparedModel?

    public init() {}

    /// Build (if needed) and return the compiled model configuration. Cached
    /// on first success; idempotent on subsequent calls. Throws a
    /// SpeechError on failure so callers can map to the standard HTTP
    /// status taxonomy.
    public func prepare(locale: Locale = Locale(identifier: "en-US")) async throws -> PreparedModel {
        if let existing = cached {
            return existing
        }

        let cacheRoot = try Self.cacheDirectory()
        let assetURL = cacheRoot.appendingPathComponent("asset.bin")
        let lmURL = cacheRoot.appendingPathComponent("lm.bin")
        let vocabURL = cacheRoot.appendingPathComponent("vocab.bin")

        // Always rebuild the asset (cheap; tens of ms) so any change to
        // the in-source phrase list takes effect on next startup. Compile
        // step honors the SDK's own cache via `ignoresCache: false`, so
        // unchanged assets skip recompilation.
        let data = SFCustomLanguageModelData(
            locale: locale,
            identifier: Self.modelIdentifier,
            version: Self.modelVersion
        )
        for phrase in Self.techPhrases {
            data.insert(phraseCount: SFCustomLanguageModelData.PhraseCount(
                phrase: phrase,
                count: Self.phraseCount
            ))
        }
        do {
            try await data.export(to: assetURL)
        } catch {
            throw SpeechError.recognitionFailed(
                "Custom language model export failed: \(error.localizedDescription)"
            )
        }

        let configuration = SFSpeechLanguageModel.Configuration(
            languageModel: lmURL,
            vocabulary: vocabURL
        )

        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                SFSpeechLanguageModel.prepareCustomLanguageModel(
                    for: assetURL,
                    configuration: configuration,
                    ignoresCache: false
                ) { error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume()
                    }
                }
            }
        } catch {
            throw SpeechError.recognitionFailed(
                "Custom language model compile failed: \(error.localizedDescription)"
            )
        }

        let prepared = PreparedModel(configuration: configuration)
        cached = prepared
        return prepared
    }

    private static func cacheDirectory() throws -> URL {
        let fm = FileManager.default
        let root = try fm.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = root
            .appendingPathComponent("com.macafm")
            .appendingPathComponent("clm")
            .appendingPathComponent(modelIdentifier)
            .appendingPathComponent(modelVersion)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Phrase set

    /// Per-phrase count in the training data. Apple's docs are coy about
    /// the exact mapping from count to bias weight but higher counts =
    /// stronger preference. 100 is a starting point; revisit after
    /// benchmarking on tech-database.wav.
    private static let phraseCount = 100

    /// Phrases that `SpeechTranscriber.contextualStrings` consistently
    /// fails to recover on the benchmark corpus. The product names here
    /// are the load-bearing ones; surrounding-sentence templates would
    /// give the LM more grammatical context but the asset format also
    /// supports raw phrase counts so we start there.
    private static let techPhrases: [String] = [
        // Containers / cloud orchestration — "Hubernet's" / "Huber needs"
        "Kubernetes",
        "Kubernetes orchestrates containerized microservices",
        "containerized microservices",
        "microservices",
        "namespaces",
        "network namespaces",
        "self-healing",
        "reconciliation loops",
        "reconciliation",
        "control plane",
        "kubelet",
        "kubectl",
        "etcd",

        // Databases — "Poster SL" / "PostGerSQL", "kilocops" / "atomisited"
        "PostgreSQL",
        "PostgreSQL stores rows in heap pages",
        "B-tree indexes",
        "B-tree",
        "ACID transactions",
        "ACID",
        "atomicity",
        "write ahead logging",
        "key lookups",
        "primary key lookups",
        "query planner",
        "selectivity estimates",
        "statistics collector",
        "sequential scan",
        "index scan",
        "bitmap heap scan",

        // AWS — "Sage maker" / "lamb of" / "elastic cash" / "incognito"
        "SageMaker",
        "DynamoDB",
        "Lambda",
        "Lambda functions",
        "Kinesis",
        "CloudFront",
        "ElastiCache",
        "Cognito",
        "Route 53",
        "API Gateway",
        "EventBridge",
        "Step Functions",
        "CloudWatch",
        "CloudTrail",

        // React / frontend — "you stayed" → useState
        "useState",
        "useEffect",
        "useMemo",
        "useCallback",
        "useRef",
        "useContext",
        "useReducer",
        "JSX",
        "virtual DOM",
        "React renders components",
        "Tailwind",
        "Tailwind utility classes",

        // Rust — "trade system" → trait system
        "borrow checker",
        "trait system",
        "trait bounds",
        "lifetimes",
        "tokio",
        "tokio runtime",
        "async functions",
        "garbage collector",

        // General DevX terms that round out the technical-English class
        "TypeScript",
        "JavaScript",
        "Python",
        "Rust",
        "Go",
        "Swift",
        "OpenAI",
        "Anthropic",
        "Claude",
        "GitHub",
        "GitLab",
        "Docker",
        "Helm",
        "Istio",
        "Prometheus",
        "Grafana",
        "Terraform",
        "Ansible",
    ]
}
