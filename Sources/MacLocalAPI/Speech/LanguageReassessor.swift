import Foundation

/// Decides whether the first transcription pass looks "clearly wrong" and a
/// retry with a different locale is worthwhile.
///
/// The trigger rule requires **all three** signals to fire before a retry is
/// proposed:
///
/// 1. Mean early-window confidence below `confidenceThreshold`.
/// 2. Out-of-vocabulary ratio above `oovRatioThreshold`.
/// 3. A non-English language guess from the lightweight identifier.
///
/// All three conjuncts keep false positives low: technical English will hit
/// (1) but fail (2), heavily accented English will hit (2) but fail (3).
/// Requiring all three means the retry path stays dormant on the common case.
struct LanguageReassessor: Sendable {
    /// Minimum mean early-window confidence below which we consider the pass
    /// "low confidence." Starting value; calibrated against the expanded
    /// corpus during implementation.
    let confidenceThreshold: Double

    /// Maximum OOV ratio above which we consider the pass "mostly unknown
    /// words." Starting value; calibrated against the expanded corpus.
    let oovRatioThreshold: Double

    /// Minimum audio duration in seconds to consider a retry. Audio shorter
    /// than this doesn't produce enough signal for the trigger rule.
    let minDurationSec: Double

    /// Starting/default values. Subject to Task 9 Step 8 calibration.
    init(
        confidenceThreshold: Double = 0.55,
        oovRatioThreshold: Double = 0.60,
        minDurationSec: Double = 1.5
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.oovRatioThreshold = oovRatioThreshold
        self.minDurationSec = minDurationSec
    }

    /// Given a first-pass `TranscriptionAttempt`, return the detected locale
    /// to retry under, or `nil` if no retry is warranted.
    func shouldRetry(
        attempt: TranscriptionAttempt,
        callerSuppliedLocale: Bool,
        audioDurationSec: Double
    ) -> Locale? {
        // Disabling conditions — return early, no retry.
        if callerSuppliedLocale { return nil }
        if audioDurationSec < minDurationSec { return nil }
        if attempt.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return nil }

        // Trigger conjunction.
        guard attempt.meanEarlyConfidence < confidenceThreshold else { return nil }
        guard attempt.oovRatio > oovRatioThreshold else { return nil }
        guard let guess = attempt.detectedLanguageGuess else { return nil }

        // "Not English" check. Compare on language code alone (en-US, en-GB, etc.).
        let guessLang = guess.language.languageCode?.identifier ?? guess.identifier
        if guessLang.hasPrefix("en") { return nil }

        return guess
    }
}

/// Out-of-vocabulary ratio calculator.
///
/// Given a token list and the set of "known" words (active contextual vocab
/// ∪ bundled English frequency list), computes the fraction of tokens that
/// are *not* known. Case-insensitive.
enum OOVCalculator {
    /// Returns `oov_count / total` in [0, 1]. Returns 0 for empty input.
    static func oovRatio(tokens: [String], known: Set<String>) -> Double {
        guard !tokens.isEmpty else { return 0 }
        var oov = 0
        for token in tokens {
            let normalized = token.lowercased()
            if !known.contains(normalized) {
                oov += 1
            }
        }
        return Double(oov) / Double(tokens.count)
    }

    /// Split a transcription string into tokens on whitespace + punctuation.
    static func tokenize(_ text: String) -> [String] {
        // Split on any whitespace or punctuation.
        let separators = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
        return text.components(separatedBy: separators).filter { !$0.isEmpty }
    }
}

/// Lightweight language identifier.
///
/// Uses character-class ratios to handle the non-Latin cases (Chinese,
/// Japanese, Korean, Cyrillic, Arabic) reliably, and a small marker-words
/// heuristic for distinguishing Latin-script languages. Not a general-purpose
/// classifier; good enough for the "is this almost certainly not English?"
/// check that the reassessor needs.
enum CharacterClassLanguageID {
    /// Return a best-guess locale from the given text, or `nil` when the text
    /// has too little signal or appears to be English.
    static func identify(text: String) -> Locale? {
        guard !text.isEmpty else { return nil }

        let counts = classCharacterCounts(text)
        let totalClassified = counts.values.reduce(0, +)
        guard totalClassified > 0 else { return nil }

        // Non-Latin scripts: if more than 30% of classified characters fall in
        // a non-Latin script, return that language.
        let nonLatinThreshold = Double(totalClassified) * 0.3
        if Double(counts[.cjkIdeograph] ?? 0) >= nonLatinThreshold {
            // Bias: unless we also see Hiragana/Katakana (Japanese) or Hangul
            // (Korean), default to Chinese.
            if (counts[.hiragana] ?? 0) + (counts[.katakana] ?? 0) > 0 {
                return Locale(identifier: "ja-JP")
            }
            if (counts[.hangul] ?? 0) > 0 {
                return Locale(identifier: "ko-KR")
            }
            return Locale(identifier: "zh-CN")
        }
        if Double(counts[.hangul] ?? 0) >= nonLatinThreshold {
            return Locale(identifier: "ko-KR")
        }
        if Double(counts[.hiragana] ?? 0) + Double(counts[.katakana] ?? 0) >= nonLatinThreshold {
            return Locale(identifier: "ja-JP")
        }
        if Double(counts[.cyrillic] ?? 0) >= nonLatinThreshold {
            return Locale(identifier: "ru-RU")
        }
        if Double(counts[.arabic] ?? 0) >= nonLatinThreshold {
            return Locale(identifier: "ar")
        }

        // Latin-script: marker-words heuristic. Counts how many high-frequency
        // function words from each candidate language appear in the text.
        let lower = text.lowercased()
        let scores: [(Locale, Int)] = [
            (Locale(identifier: "es-ES"), markerHits(lower, markers: Self.spanishMarkers)),
            (Locale(identifier: "fr-FR"), markerHits(lower, markers: Self.frenchMarkers)),
            (Locale(identifier: "de-DE"), markerHits(lower, markers: Self.germanMarkers)),
            (Locale(identifier: "it-IT"), markerHits(lower, markers: Self.italianMarkers)),
            (Locale(identifier: "pt-PT"), markerHits(lower, markers: Self.portugueseMarkers))
        ]
        let englishScore = markerHits(lower, markers: Self.englishMarkers)
        let bestNonEnglish = scores.max(by: { $0.1 < $1.1 })

        guard let candidate = bestNonEnglish, candidate.1 > englishScore, candidate.1 >= 2 else {
            return nil
        }
        return candidate.0
    }

    // MARK: - Internals

    enum CharacterClass: Hashable {
        case latin
        case cyrillic
        case arabic
        case cjkIdeograph
        case hiragana
        case katakana
        case hangul
        case other
    }

    static func classCharacterCounts(_ text: String) -> [CharacterClass: Int] {
        var counts: [CharacterClass: Int] = [:]
        for scalar in text.unicodeScalars {
            let c = classify(scalar)
            guard c != .other else { continue }
            counts[c, default: 0] += 1
        }
        return counts
    }

    private static func classify(_ scalar: Unicode.Scalar) -> CharacterClass {
        let v = scalar.value
        switch v {
        case 0x0041...0x007A:            // A-Z, a-z (with small gap for [\]^_`)
            if (0x005B...0x0060).contains(v) { return .other }
            return .latin
        case 0x00C0...0x024F:             // Latin Extended
            return .latin
        case 0x0400...0x04FF:             // Cyrillic
            return .cyrillic
        case 0x0600...0x06FF:             // Arabic
            return .arabic
        case 0x3040...0x309F:             // Hiragana
            return .hiragana
        case 0x30A0...0x30FF:             // Katakana
            return .katakana
        case 0xAC00...0xD7AF:             // Hangul syllables
            return .hangul
        case 0x3400...0x9FFF:             // CJK Unified Ideographs (incl ext A)
            return .cjkIdeograph
        default:
            return .other
        }
    }

    private static func markerHits(_ lowerText: String, markers: [String]) -> Int {
        var hits = 0
        for m in markers {
            // Count whole-word-ish hits to avoid "el" matching inside "help".
            let padded = " \(m) "
            let paddedText = " \(lowerText) "
            if paddedText.contains(padded) {
                hits += 1
            }
        }
        return hits
    }

    // High-frequency function words per language. Small curated lists —
    // not a statistical model, but sufficient to tell "mostly English" from
    // "mostly Spanish" on a few seconds of transcribed text.
    static let englishMarkers: [String] = [
        "the", "and", "of", "to", "is", "that", "it", "for", "with", "this",
        "you", "not", "but", "are", "was", "have", "be", "as", "on", "at"
    ]
    static let spanishMarkers: [String] = [
        "el", "la", "los", "las", "que", "de", "y", "es", "en", "por",
        "para", "con", "un", "una", "del", "se", "lo", "no", "más", "pero"
    ]
    static let frenchMarkers: [String] = [
        "le", "la", "les", "de", "des", "et", "est", "en", "un", "une",
        "que", "pour", "dans", "sur", "avec", "pas", "mais", "à", "au", "aux"
    ]
    static let germanMarkers: [String] = [
        "der", "die", "das", "und", "ist", "nicht", "mit", "ein", "eine", "zu",
        "auch", "auf", "sich", "für", "von", "den", "dem", "nur", "bei", "aus"
    ]
    static let italianMarkers: [String] = [
        "il", "lo", "la", "gli", "le", "di", "e", "è", "per", "con",
        "un", "una", "che", "non", "ma", "ci", "si", "come", "dal", "sul"
    ]
    static let portugueseMarkers: [String] = [
        "o", "a", "os", "as", "de", "e", "é", "que", "para", "com",
        "um", "uma", "do", "da", "dos", "das", "não", "mas", "por", "mais"
    ]
}
