import AVFoundation
import CoreMedia
import Foundation
import Speech
import XCTest

@testable import MacLocalAPI

/// Unit tests for the pure-logic parts of `SpeechTranscriberEngine`.
///
/// Runtime integration tests that exercise `SpeechAnalyzer` require Speech
/// Recognition TCC consent (granted once at OS level) and one-time locale
/// model downloads, so they live in `SpeechTranscriberEngineIntegrationTests`
/// which is gated on a test-host fixture file existing.
final class SpeechTranscriberEngineTests: XCTestCase {

    // MARK: - cmTimeMs

    func testCMTimeMsRoundsToMilliseconds() {
        let t = CMTime(seconds: 1.234, preferredTimescale: 1_000)
        XCTAssertEqual(SpeechTranscriberEngine.cmTimeMs(t), 1234)
    }

    func testCMTimeMsHandlesZero() {
        let t = CMTime.zero
        XCTAssertEqual(SpeechTranscriberEngine.cmTimeMs(t), 0)
    }

    // MARK: - assembleAttempt on empty results

    func testAssembleEmptyResultsYieldsHighConfidenceNoRetrySignals() {
        let attempt = SpeechTranscriberEngine.assembleAttempt(
            from: [],
            durationSec: 10,
            activeVocab: Set<String>()
        )
        XCTAssertEqual(attempt.text, "")
        // No signal → meanEarlyConfidence defaults to 1.0 so the reassessor
        // doesn't fire spuriously on silent audio.
        XCTAssertEqual(attempt.meanEarlyConfidence, 1.0)
        XCTAssertEqual(attempt.oovRatio, 0.0)
        XCTAssertNil(attempt.detectedLanguageGuess)
    }

    // MARK: - Pipeline smoke via integration test

    /// Runtime integration test. Requires:
    ///  - Speech Recognition TCC grant for the test runner
    ///  - macOS 26 with en-US model installed (first use may trigger download)
    ///  - A WAV at Scripts/test-data/speech/short-5s.wav
    ///
    /// Skipped with a clear message when the fixture is missing; this keeps
    /// the suite green on machines where the corpus has not been generated.
    func testTranscribesShortEnglishClip() async throws {
        guard let fixtureURL = Self.speechFixtureURL("short-5s.wav") else {
            throw XCTSkip("short-5s.wav fixture not found; skip runtime transcription test")
        }
        let engine = SpeechTranscriberEngine()
        let attempt = try await engine.transcribe(
            url: fixtureURL,
            locale: Locale(identifier: "en-US"),
            contextualStrings: ["Kubernetes"],  // arbitrary; fixture is generic english
            wantWordTimings: true
        )
        XCTAssertFalse(attempt.text.isEmpty, "Expected non-empty transcription")
        // Sanity: at least some word timings should be emitted under the
        // time-indexed progressive preset.
        XCTAssertFalse(attempt.words.isEmpty, "Expected word timings with timeIndexedProgressiveTranscription preset")
        // Confidence should be high for a clean TTS-style clip.
        XCTAssertGreaterThan(attempt.meanEarlyConfidence, 0.5)
    }

    // MARK: - Helpers

    /// Resolve a Scripts/test-data/speech/<name> path relative to the repo
    /// root. Navigates up from #filePath (which points at this test file).
    private static func speechFixtureURL(_ filename: String) -> URL? {
        let thisFile = URL(fileURLWithPath: #filePath)
        // Tests/MacLocalAPITests/Speech/SpeechTranscriberEngineTests.swift
        //  → ../../../Scripts/test-data/speech/<filename>
        let repoRoot = thisFile
            .deletingLastPathComponent()  // Speech
            .deletingLastPathComponent()  // MacLocalAPITests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let fixture = repoRoot
            .appendingPathComponent("Scripts")
            .appendingPathComponent("test-data")
            .appendingPathComponent("speech")
            .appendingPathComponent(filename)
        return FileManager.default.fileExists(atPath: fixture.path) ? fixture : nil
    }
}
