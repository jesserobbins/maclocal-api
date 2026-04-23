import Foundation
import XCTest

@testable import MacLocalAPI

final class SpeechPipelineServiceTests: XCTestCase {

    func testMakeDefaultConstructsStack() async throws {
        // This builds the full default pipeline (preprocessor, resolver from
        // the bundled vocab, pool with en-US warm, reassessor). Runs in
        // the unit-test target without TCC.
        let svc = try await SpeechPipelineService.makeDefault()
        XCTAssertNotNil(svc)
    }

    func testPipelineOptionsDefaults() {
        let opts = PipelineRequestOptions()
        XCTAssertEqual(opts.locale, "en-US")
        XCTAssertNil(opts.prompt)
        XCTAssertTrue(opts.wantWordTimings)
        XCTAssertFalse(opts.callerSuppliedLocale)
    }

    func testPipelineOptionsRespectsCallerSupplied() {
        let opts = PipelineRequestOptions(
            locale: "es-ES",
            prompt: "tachyon",
            wantWordTimings: false,
            callerSuppliedLocale: true
        )
        XCTAssertEqual(opts.locale, "es-ES")
        XCTAssertEqual(opts.prompt, "tachyon")
        XCTAssertFalse(opts.wantWordTimings)
        XCTAssertTrue(opts.callerSuppliedLocale)
    }

    /// Runtime integration: exercises the full pipeline including TCC-gated
    /// SpeechAnalyzer. Skipped when no fixture is available in the worktree.
    func testPipelineEndToEndOnFixture() async throws {
        guard let url = Self.speechFixtureURL() else {
            throw XCTSkip("No speech fixture available; skip pipeline integration test")
        }
        let svc = try await SpeechPipelineService.makeDefault()
        let result = try await svc.transcribe(
            url: url,
            options: PipelineRequestOptions(locale: "en-US", callerSuppliedLocale: true)
        )
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertEqual(result.language, "en-US")
    }

    private static func speechFixtureURL() -> URL? {
        let thisFile = URL(fileURLWithPath: #filePath)
        let repoRoot = thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            "short-5s.wav",
            "clean-narration.wav",
            "fast-speech.wav"
        ]
        for name in candidates {
            let u = repoRoot
                .appendingPathComponent("Scripts")
                .appendingPathComponent("test-data")
                .appendingPathComponent("speech")
                .appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: u.path) {
                return u
            }
        }
        return nil
    }
}
