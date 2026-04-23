import XCTest

@testable import MacLocalAPI

final class SpeechTypesTests: XCTestCase {
    func testSpeechResponseFormatRawValuesMatchOpenAIWire() {
        XCTAssertEqual(SpeechResponseFormat.text.rawValue, "text")
        XCTAssertEqual(SpeechResponseFormat.json.rawValue, "json")
        XCTAssertEqual(SpeechResponseFormat.verboseJson.rawValue, "verbose_json")
        XCTAssertEqual(SpeechResponseFormat.srt.rawValue, "srt")
        XCTAssertEqual(SpeechResponseFormat.vtt.rawValue, "vtt")
    }

    func testTimestampGranularityDefaultsToSegment() {
        let granularity: TimestampGranularity = .segment
        XCTAssertEqual(granularity.rawValue, "segment")
    }

    func testWordTimingMonotonicWithinSegment() {
        let w1 = WordTiming(word: "hello", startMs: 0, endMs: 400, confidence: 0.95)
        let w2 = WordTiming(word: "world", startMs: 400, endMs: 820, confidence: 0.88)
        XCTAssertLessThan(w1.endMs, w2.endMs)
        XCTAssertLessThanOrEqual(w1.endMs, w2.startMs)
    }

    func testTranscriptionResultVerboseShape() {
        let words = [
            WordTiming(word: "hello", startMs: 0, endMs: 400, confidence: 0.95),
            WordTiming(word: "world", startMs: 400, endMs: 820, confidence: 0.88)
        ]
        let seg = Segment(text: "hello world", startMs: 0, endMs: 820, meanConfidence: 0.915)
        let result = TranscriptionResult(
            text: "hello world",
            language: "en-US",
            durationMs: 820,
            segments: [seg],
            words: words,
            languageReassessed: false
        )
        XCTAssertEqual(result.segments?.count, 1)
        XCTAssertEqual(result.words?.count, 2)
        XCTAssertFalse(result.languageReassessed)
    }

    func testTranscriptionAttemptSignalsPresent() {
        let attempt = TranscriptionAttempt(
            text: "hello",
            meanEarlyConfidence: 0.9,
            oovRatio: 0.1,
            segments: [],
            words: [],
            detectedLanguageGuess: nil
        )
        XCTAssertEqual(attempt.meanEarlyConfidence, 0.9, accuracy: 0.001)
    }
}
