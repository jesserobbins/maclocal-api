import XCTest

@testable import MacLocalAPI

final class VoiceActivityTrimmerTests: XCTestCase {
    private let sampleRate: Double = 16_000

    func testAllSilenceIsLeftAlone() {
        // 2s of pure zeros — no peak above threshold. Trimmer should not trim
        // (defers to the transcriber to handle truly silent input).
        let samples = [Float](repeating: 0, count: 32_000)
        let result = VoiceActivityTrimmer.trim(samples)
        XCTAssertFalse(result.didTrim)
        XCTAssertEqual(result.samples.count, samples.count)
        XCTAssertEqual(result.leadingTrimmedSamples, 0)
        XCTAssertEqual(result.trailingTrimmedSamples, 0)
    }

    func testFullyVoicedAudioIsLeftAlone() {
        // 2s of a steady sine — every window is active, nothing to trim.
        let samples = sineWave(durationSec: 2.0, amplitude: 0.1)
        let result = VoiceActivityTrimmer.trim(samples)
        XCTAssertFalse(result.didTrim)
        XCTAssertEqual(result.samples.count, samples.count)
    }

    func testLeadingSilenceIsTrimmed() {
        // 1s silence + 2s sine. Expect trim of ~1s minus the hangover pad.
        let silence = [Float](repeating: 0, count: 16_000)
        let voice = sineWave(durationSec: 2.0, amplitude: 0.1)
        let samples = silence + voice
        let result = VoiceActivityTrimmer.trim(samples)

        XCTAssertTrue(result.didTrim)
        XCTAssertEqual(result.trailingTrimmedSamples, 0)
        // Leading trim should be roughly 1s minus 100ms hangover = ~14_400.
        // Allow ±1 window of slack (320 samples).
        XCTAssertEqual(
            result.leadingTrimmedSamples,
            16_000 - VoiceActivityTrimmer.hangoverSamples,
            accuracy: VoiceActivityTrimmer.windowSamples
        )
        XCTAssertEqual(
            result.samples.count,
            samples.count - result.leadingTrimmedSamples
        )
    }

    func testTrailingSilenceIsTrimmed() {
        let voice = sineWave(durationSec: 2.0, amplitude: 0.1)
        let silence = [Float](repeating: 0, count: 16_000)
        let samples = voice + silence
        let result = VoiceActivityTrimmer.trim(samples)

        XCTAssertTrue(result.didTrim)
        XCTAssertEqual(result.leadingTrimmedSamples, 0)
        XCTAssertEqual(
            result.trailingTrimmedSamples,
            16_000 - VoiceActivityTrimmer.hangoverSamples,
            accuracy: VoiceActivityTrimmer.windowSamples
        )
    }

    func testLeadingAndTrailingSilenceBothTrimmed() {
        let lead = [Float](repeating: 0, count: 16_000)
        let voice = sineWave(durationSec: 2.0, amplitude: 0.1)
        let trail = [Float](repeating: 0, count: 16_000)
        let samples = lead + voice + trail
        let result = VoiceActivityTrimmer.trim(samples)

        XCTAssertTrue(result.didTrim)
        XCTAssertGreaterThan(result.leadingTrimmedSamples, 0)
        XCTAssertGreaterThan(result.trailingTrimmedSamples, 0)
        // Output should be roughly the voice + 2x hangover.
        let expected = voice.count + 2 * VoiceActivityTrimmer.hangoverSamples
        XCTAssertEqual(result.samples.count, expected, accuracy: 2 * VoiceActivityTrimmer.windowSamples)
    }

    func testInternalSilenceIsPreserved() {
        // voice + 2s internal silence + voice. We do leading/trailing only,
        // so the middle silence must survive.
        let voice1 = sineWave(durationSec: 1.0, amplitude: 0.1)
        let middle = [Float](repeating: 0, count: 32_000)
        let voice2 = sineWave(durationSec: 1.0, amplitude: 0.1)
        let samples = voice1 + middle + voice2
        let result = VoiceActivityTrimmer.trim(samples)

        // No leading/trailing silence, so no trim.
        XCTAssertFalse(result.didTrim)
        XCTAssertEqual(result.samples.count, samples.count)
    }

    func testVerySmallTrimIsSkipped() {
        // 50ms of leading silence — below the 200ms minimum trim threshold.
        let lead = [Float](repeating: 0, count: 800)
        let voice = sineWave(durationSec: 2.0, amplitude: 0.1)
        let samples = lead + voice
        let result = VoiceActivityTrimmer.trim(samples)
        XCTAssertFalse(result.didTrim)
        XCTAssertEqual(result.samples.count, samples.count)
    }

    func testShorterThanWindowIsLeftAlone() {
        let samples = [Float](repeating: 0.05, count: 100)
        let result = VoiceActivityTrimmer.trim(samples)
        XCTAssertFalse(result.didTrim)
        XCTAssertEqual(result.samples.count, samples.count)
    }

    // MARK: - Helpers

    private func sineWave(durationSec: Double, amplitude: Float) -> [Float] {
        let count = Int(durationSec * sampleRate)
        let twoPi: Float = 2.0 * .pi
        let freq: Float = 440.0
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let t = Float(i) / Float(sampleRate)
            out[i] = amplitude * sin(twoPi * freq * t)
        }
        return out
    }
}
