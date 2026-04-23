import AVFoundation
import XCTest

@testable import MacLocalAPI

final class AudioPreprocessorTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() async throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AudioPreprocessorTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testAlreadyMatchedFormatSkipsResample() async throws {
        let url = try synthWAV(
            at: tempDir.appendingPathComponent("matched.wav"),
            sampleRate: 16_000,
            channels: 1,
            durationSec: 1.0,
            amplitude: 0.1
        )

        let pp = AudioPreprocessor()
        let prepared = try await pp.prepare(url: url)

        XCTAssertEqual(prepared.sampleRate, 16_000)
        XCTAssertFalse(prepared.wasResampled, "Matched format must not be resampled")
        XCTAssertGreaterThan(prepared.durationMs, 900)
        XCTAssertLessThan(prepared.durationMs, 1_100)
    }

    func testMismatchedFormatIsResampled() async throws {
        let url = try synthWAV(
            at: tempDir.appendingPathComponent("mismatched.wav"),
            sampleRate: 44_100,
            channels: 2,
            durationSec: 0.5,
            amplitude: 0.1
        )

        let pp = AudioPreprocessor()
        let prepared = try await pp.prepare(url: url)

        XCTAssertEqual(prepared.sampleRate, 16_000)
        XCTAssertTrue(prepared.wasResampled)
    }

    func testQuietAudioIsNormalized() async throws {
        // RMS ~0.003 → about -50 dBFS; well below the -29 dBFS acceptable floor.
        let url = try synthWAV(
            at: tempDir.appendingPathComponent("quiet.wav"),
            sampleRate: 16_000,
            channels: 1,
            durationSec: 1.0,
            amplitude: 0.003
        )

        let pp = AudioPreprocessor()
        let prepared = try await pp.prepare(url: url)

        XCTAssertFalse(prepared.wasResampled)
        XCTAssertTrue(prepared.wasLoudnessNormalized, "Very quiet audio should be normalized")
    }

    func testWellMixedAudioIsNotNormalized() async throws {
        // RMS ~0.07 → about -23 dBFS, squarely inside the acceptable band.
        let url = try synthWAV(
            at: tempDir.appendingPathComponent("mixed.wav"),
            sampleRate: 16_000,
            channels: 1,
            durationSec: 1.0,
            amplitude: 0.1  // sine wave peak 0.1 → RMS ~0.0707
        )

        let pp = AudioPreprocessor()
        let prepared = try await pp.prepare(url: url)

        XCTAssertFalse(prepared.wasLoudnessNormalized, "Within-band audio should be left alone")
    }

    func testPreparedAudioEmitsMultipleBuffers() async throws {
        let url = try synthWAV(
            at: tempDir.appendingPathComponent("long.wav"),
            sampleRate: 16_000,
            channels: 1,
            durationSec: 4.0,  // 4 s at 16 kHz = 64k samples / 16k chunk = 4 chunks
            amplitude: 0.1
        )

        let pp = AudioPreprocessor()
        let prepared = try await pp.prepare(url: url)

        var chunkCount = 0
        var totalSamples = 0
        for await chunk in prepared.stream {
            chunkCount += 1
            totalSamples += chunk.samples.count
        }
        XCTAssertGreaterThanOrEqual(chunkCount, 3, "Expected multiple chunks for a 4s file")
        XCTAssertGreaterThan(totalSamples, 60_000)
    }

    // MARK: - Helpers

    /// Synthesize a sine-wave WAV at the given sample rate + channels + duration.
    /// Amplitude is the peak value; RMS = amplitude / sqrt(2).
    private func synthWAV(
        at url: URL,
        sampleRate: Double,
        channels: AVAudioChannelCount,
        durationSec: Double,
        amplitude: Float
    ) throws -> URL {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        )!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)

        let frameCount = AVAudioFrameCount(durationSec * sampleRate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount

        let ptrs = buffer.floatChannelData!
        let freq: Float = 440.0
        let twoPi: Float = 2.0 * .pi
        for ch in 0..<Int(channels) {
            let chPtr = ptrs[ch]
            for f in 0..<Int(frameCount) {
                let t = Float(f) / Float(sampleRate)
                chPtr[f] = amplitude * sin(twoPi * freq * t)
            }
        }
        try file.write(from: buffer)
        return url
    }
}
