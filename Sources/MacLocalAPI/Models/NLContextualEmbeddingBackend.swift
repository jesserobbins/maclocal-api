import Foundation
import NaturalLanguage
import Accelerate

actor NLContextualEmbeddingBackend: EmbeddingBackend {
    private static let multilingualScript = NLScript.latin

    let modelID: String
    let nativeDimension: Int
    let maxInputTokens: Int

    private let embedding: NLContextualEmbedding
    private let language: NLLanguage?
    private var isLoaded = false

    init(modelID: String) throws {
        guard let selection = Self.selection(for: modelID) else {
            throw EmbeddingError.modelNotFound(modelID)
        }

        self.modelID = modelID
        self.embedding = selection.embedding
        self.language = selection.language
        self.nativeDimension = Int(selection.embedding.dimension)
        self.maxInputTokens = Int(selection.embedding.maximumSequenceLength)
    }

    func prepare() async throws {
        if isLoaded {
            return
        }
        try await ensureAssetsAvailable()
        try loadIfNeeded()
    }

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one input is required")
        }

        try await prepare()

        var vectors: [[Float]] = []
        var tokenCounts: [Int] = []
        var truncatedInputCount = 0

        for input in inputs {
            guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw EmbeddingError.invalidInput("Input strings must not be empty or whitespace-only")
            }

            let result: NLContextualEmbeddingResult
            do {
                result = try embedding.embeddingResult(for: input, language: language)
            } catch let embeddingError as EmbeddingError {
                throw embeddingError
            } catch {
                throw EmbeddingError.backendUnavailable(
                    id: modelID,
                    reason: error.localizedDescription
                )
            }

            let sequenceLength = Int(result.sequenceLength)
            tokenCounts.append(sequenceLength)
            // NLContextualEmbedding silently truncates inputs to maximumSequenceLength.
            // When the returned sequence length equals the backend cap, treat it as
            // truncated so clients get an X-Embedding-Truncated signal. This over-
            // counts inputs that happen to land exactly at the cap, but under-
            // counting (the previous behavior — always 0) is worse for the
            // long-document workflows this header exists for.
            if sequenceLength >= maxInputTokens {
                truncatedInputCount += 1
            }
            vectors.append(try poolMeanNormalized(result: result))
        }

        return EmbedResult(
            vectors: vectors,
            tokenCounts: tokenCounts,
            truncatedInputCount: truncatedInputCount
        )
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        guard !inputs.isEmpty else {
            throw EmbeddingError.invalidInput("At least one token-id input is required")
        }
        throw EmbeddingError.invalidInput("Pre-tokenized input is not supported by Apple NL embeddings")
    }

    private func ensureAssetsAvailable() async throws {
        guard !embedding.hasAvailableAssets else {
            return
        }

        print("[Embeddings] Requesting Apple NL assets for \(modelID)...")

        do {
            let result = try await embedding.requestAssets()
            switch result {
            case .available:
                print("[Embeddings] Apple NL assets ready for \(modelID).")
            case .notAvailable:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "Assets were not available for download"
                )
            case .error:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "The NaturalLanguage framework reported an asset download error"
                )
            @unknown default:
                throw EmbeddingError.assetDownloadFailed(
                    id: modelID,
                    reason: "The NaturalLanguage framework returned an unknown asset result"
                )
            }
        } catch let embeddingError as EmbeddingError {
            throw embeddingError
        } catch {
            throw EmbeddingError.assetDownloadFailed(id: modelID, reason: error.localizedDescription)
        }
    }

    private func loadIfNeeded() throws {
        guard !isLoaded else {
            return
        }

        do {
            try embedding.load()
            isLoaded = true
        } catch {
            throw EmbeddingError.backendUnavailable(id: modelID, reason: error.localizedDescription)
        }
    }

    private func poolMeanNormalized(result: NLContextualEmbeddingResult) throws -> [Float] {
        // Sum per-token vectors in Double to match NLContextualEmbedding's
        // native [Double] output (no per-token Float conversion), then convert
        // and divide once at the end. vDSP_vaddD is a SIMD accumulation,
        // replacing the scalar `sum[index] += Float(value)` loop that ran
        // dim*tokens times per input.
        let dim = nativeDimension
        let n = vDSP_Length(dim)
        var doubleSum = [Double](repeating: 0, count: dim)
        var tokenCount = 0
        let fullRange = result.string.startIndex..<result.string.endIndex

        doubleSum.withUnsafeMutableBufferPointer { sumPtr in
            guard let sumBase = sumPtr.baseAddress else { return }
            result.enumerateTokenVectors(in: fullRange) { tokenVector, _ in
                guard tokenVector.count == dim else {
                    return true
                }
                tokenVector.withUnsafeBufferPointer { tokPtr in
                    guard let tokBase = tokPtr.baseAddress else { return }
                    vDSP_vaddD(sumBase, 1, tokBase, 1, sumBase, 1, n)
                }
                tokenCount += 1
                return true
            }
        }

        guard tokenCount > 0 else {
            throw EmbeddingError.internalFailure
        }

        var scaleD = Double(tokenCount)
        var meanD = [Double](repeating: 0, count: dim)
        vDSP_vsdivD(doubleSum, 1, &scaleD, &meanD, 1, n)
        var meanF = [Float](repeating: 0, count: dim)
        vDSP_vdpsp(meanD, 1, &meanF, 1, n)
        return EmbeddingMath.l2Normalize(meanF)
    }

    private static func selection(for modelID: String) -> (embedding: NLContextualEmbedding, language: NLLanguage?)? {
        switch modelID {
        case EmbeddingModelRegistry.defaultModelID:
            guard let embedding = NLContextualEmbedding(language: NLLanguage.english) else {
                return nil
            }
            return (embedding, NLLanguage.english)
        case EmbeddingModelRegistry.multilingualModelID:
            guard let embedding = NLContextualEmbedding(script: multilingualScript) else {
                return nil
            }
            return (embedding, nil)
        default:
            return nil
        }
    }
}
