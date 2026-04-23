import XCTest

@testable import MacLocalAPI

final class LanguageReassessorTests: XCTestCase {

    // MARK: - Trigger conjunction

    private func attempt(
        text: String = "some transcribed english words here",
        conf: Double = 0.9,
        oov: Double = 0.1,
        guess: Locale? = nil
    ) -> TranscriptionAttempt {
        TranscriptionAttempt(
            text: text,
            meanEarlyConfidence: conf,
            oovRatio: oov,
            segments: [],
            words: [],
            detectedLanguageGuess: guess
        )
    }

    func testNoRetryWhenCallerSuppliedLocale() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.9, guess: Locale(identifier: "es-ES"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: true, audioDurationSec: 10))
    }

    func testNoRetryWhenBelowMinDuration() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.9, guess: Locale(identifier: "es-ES"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 1.0))
    }

    func testNoRetryWhenTextEmpty() {
        let r = LanguageReassessor()
        let a = attempt(text: "   ", conf: 0.3, oov: 0.9, guess: Locale(identifier: "es-ES"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10))
    }

    func testNoRetryWhenConfidenceFine() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.9, oov: 0.9, guess: Locale(identifier: "es-ES"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10))
    }

    func testNoRetryWhenOOVLow() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.2, guess: Locale(identifier: "es-ES"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10))
    }

    func testNoRetryWhenNoLanguageGuess() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.9, guess: nil)
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10))
    }

    func testNoRetryWhenGuessIsEnglish() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.9, guess: Locale(identifier: "en-GB"))
        XCTAssertNil(r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10))
    }

    func testRetryWhenAllThreeFire() {
        let r = LanguageReassessor()
        let a = attempt(conf: 0.3, oov: 0.9, guess: Locale(identifier: "es-ES"))
        let retry = r.shouldRetry(attempt: a, callerSuppliedLocale: false, audioDurationSec: 10)
        XCTAssertNotNil(retry)
        XCTAssertEqual(retry?.language.languageCode?.identifier, "es")
    }

    // MARK: - OOVCalculator

    func testOOVEmptyReturnsZero() {
        XCTAssertEqual(OOVCalculator.oovRatio(tokens: [], known: Set(["a"])), 0.0)
    }

    func testOOVAllKnown() {
        let tokens = ["The", "QUICK", "brown"]
        let known: Set<String> = ["the", "quick", "brown"]
        XCTAssertEqual(OOVCalculator.oovRatio(tokens: tokens, known: known), 0.0)
    }

    func testOOVAllUnknown() {
        XCTAssertEqual(OOVCalculator.oovRatio(tokens: ["xyz", "pqr"], known: ["abc"]), 1.0)
    }

    func testOOVMixed() {
        let tokens = ["the", "kubernetes", "and", "tachyon"]
        let known: Set<String> = ["the", "and"]
        XCTAssertEqual(OOVCalculator.oovRatio(tokens: tokens, known: known), 0.5, accuracy: 1e-9)
    }

    func testTokenizeStripsPunctuation() {
        let toks = OOVCalculator.tokenize("Hello, world! How's it-going?")
        XCTAssertTrue(toks.contains("Hello"))
        XCTAssertTrue(toks.contains("world"))
        // The apostrophe-contraction may split or not depending on punctuation
        // set; at minimum "How" and "s" or "How's" are fine. Assert the
        // content doesn't contain empty strings.
        XCTAssertFalse(toks.contains(""))
    }

    // MARK: - CharacterClassLanguageID

    func testIdentifyNonLatinChinese() {
        // Not Japanese (no kana), not Korean (no Hangul) → Chinese.
        let locale = CharacterClassLanguageID.identify(text: "这是一段中文测试文本")
        XCTAssertEqual(locale?.language.languageCode?.identifier, "zh")
    }

    func testIdentifyJapanese() {
        // Mix of hiragana and some ideographs.
        let locale = CharacterClassLanguageID.identify(text: "これは日本語のテストです")
        XCTAssertEqual(locale?.language.languageCode?.identifier, "ja")
    }

    func testIdentifyKorean() {
        let locale = CharacterClassLanguageID.identify(text: "이것은 한국어 테스트입니다")
        XCTAssertEqual(locale?.language.languageCode?.identifier, "ko")
    }

    func testIdentifySpanishViaMarkers() {
        let text = "el perro come la comida de los gatos que están en el parque"
        let locale = CharacterClassLanguageID.identify(text: text)
        XCTAssertEqual(locale?.language.languageCode?.identifier, "es")
    }

    func testIdentifyFrenchViaMarkers() {
        let text = "le chien mange la nourriture des chats qui sont dans le parc avec un ami"
        let locale = CharacterClassLanguageID.identify(text: text)
        XCTAssertEqual(locale?.language.languageCode?.identifier, "fr")
    }

    func testIdentifyEnglishReturnsNil() {
        let text = "the cat is sitting on the mat and it is not amused by this"
        XCTAssertNil(CharacterClassLanguageID.identify(text: text))
    }

    func testIdentifyTooShortReturnsNil() {
        XCTAssertNil(CharacterClassLanguageID.identify(text: "hi"))
    }
}
