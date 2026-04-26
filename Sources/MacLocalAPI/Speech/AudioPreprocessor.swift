import Accelerate
import AVFoundation
import Foundation

/// Prepares audio files for `SpeechAnalyzer` consumption.
///
/// The preprocessor is deliberately conditional: it skips work when the input
/// is already in the format the transcriber prefers (PCM 16 kHz mono f32,
/// reasonable loudness), because every stage we skip is one we can't mis-apply.
/// This matches the zero-config goal — we want to leave good input alone and
/// only intervene when intervention clearly helps.
public actor AudioPreprocessor {
    /// Target format for `SpeechTranscriber` input. 16 kHz mono float32.
    public static let targetSampleRate: Double = 16_000
    public static let targetChannels: AVAudioChannelCount = 1

    /// Emitted chunk size in samples. ~1 s at target rate keeps the
    /// `SpeechAnalyzer` fed steadily without spawning too many scheduler hops.
    public static let chunkSamples: Int = 16_000

    /// Integrated-loudness band (in dBFS RMS) that we consider "already fine".
    /// Anything inside this band is left alone; outside we apply a single gain
    /// factor to pull the RMS back toward the target.
    public static let loudnessAcceptableMinDBFS: Double = -29.0
    public static let loudnessAcceptableMaxDBFS: Double = -17.0
    public static let loudnessTargetDBFS: Double = -23.0

    public init() {}

    public func prepare(url: URL) async throws -> PreparedAudio {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat

        let needsResample = !AudioPreprocessor.isTargetFormat(sourceFormat)
        let targetFormat = needsResample ? AudioPreprocessor.targetFormat() : sourceFormat

        // Load all samples into a contiguous Float buffer at target rate.
        let samples: [Float]
        if needsResample {
            samples = try AudioPreprocessor.resample(
                file: file,
                from: sourceFormat,
                to: targetFormat
            )
        } else {
            samples = try AudioPreprocessor.readAsFloat(file: file, format: sourceFormat)
        }

        let (normalizedSamples, didNormalize) = AudioPreprocessor.loudnessNormalizeIfNeeded(samples)
        let durationMs = Int((Double(normalizedSamples.count) / AudioPreprocessor.targetSampleRate) * 1000.0)

        let stream = AudioPreprocessor.chunkedStream(samples: normalizedSamples)

        return PreparedAudio(
            stream: stream,
            durationMs: durationMs,
            sampleRate: AudioPreprocessor.targetSampleRate,
            wasResampled: needsResample,
            wasLoudnessNormalized: didNormalize
        )
    }

    // MARK: - Format inspection

    private static func isTargetFormat(_ format: AVAudioFormat) -> Bool {
        return format.sampleRate == targetSampleRate
            && format.channelCount == targetChannels
            && format.commonFormat == .pcmFormatFloat32
    }

    private static func targetFormat() -> AVAudioFormat {
        return AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: targetChannels,
            interleaved: false
        )!
    }

    // MARK: - Read + resample

    private static func readAsFloat(file: AVAudioFile, format: AVAudioFormat) throws -> [Float] {
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return []
        }
        try file.read(into: buffer)
        return floatSamples(from: buffer)
    }

    private static func resample(
        file: AVAudioFile,
        from sourceFormat: AVAudioFormat,
        to targetFormat: AVAudioFormat
    ) throws -> [Float] {
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw PreprocessorError.converterInitFailed
        }

        // Read the whole source file into a PCM buffer.
        let sourceFrameCount = AVAudioFrameCount(file.length)
        guard sourceFrameCount > 0,
              let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: sourceFrameCount) else {
            return []
        }
        try file.read(into: sourceBuffer)

        // Worst-case output frame count: source frames * rate ratio, rounded up.
        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(sourceFrameCount) * ratio + 1024)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            throw PreprocessorError.outputBufferAllocFailed
        }

        var sourceConsumed = false
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if sourceConsumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            sourceConsumed = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        if status == .error || error != nil {
            throw PreprocessorError.conversionFailed(underlying: error?.localizedDescription ?? "unknown")
        }

        return floatSamples(from: outBuffer)
    }

    private static func floatSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return [] }

        // We only need channel 0 because targetChannels == 1. For stereo input
        // that was *not* resampled (path that keeps source format), downmix by
        // averaging — but the current flow resamples whenever channels != 1,
        // so this path only fires for mono-f32 sources where channel 0 is
        // the only channel.
        if let ptrs = buffer.floatChannelData {
            return Array(UnsafeBufferPointer(start: ptrs[0], count: frames))
        }

        // Convert int16 / int32 PCM to float by normalizing to [-1, 1].
        // Only reached if the reader produced an int-format buffer — rare for
        // AVAudioFile.processingFormat which is almost always float.
        if let ptrs = buffer.int16ChannelData {
            let raw = UnsafeBufferPointer(start: ptrs[0], count: frames)
            var output = [Float](repeating: 0, count: frames)
            let n = vDSP_Length(frames)
            // vDSP_vflt16: int16 → float (cast, not scaled). Then a single
            // vDSP_vsmul scales into [-1, 1] in one SIMD pass, replacing the
            // per-element `Float($0) / Float(Int16.max)` map.
            vDSP_vflt16(raw.baseAddress!, 1, &output, 1, n)
            var scale = 1.0 / Float(Int16.max)
            vDSP_vsmul(output, 1, &scale, &output, 1, n)
            return output
        }
        return []
    }

    // MARK: - Loudness

    private static func loudnessNormalizeIfNeeded(_ samples: [Float]) -> ([Float], Bool) {
        guard !samples.isEmpty else { return (samples, false) }
        let rms = rmsDBFS(samples)
        if rms.isFinite == false {
            return (samples, false)
        }
        if rms >= loudnessAcceptableMinDBFS && rms <= loudnessAcceptableMaxDBFS {
            return (samples, false)
        }
        let gainDB = loudnessTargetDBFS - rms
        var gainLinear = Float(pow(10.0, gainDB / 20.0))
        var lowerBound: Float = -1.0
        var upperBound: Float = 1.0
        var out = [Float](repeating: 0, count: samples.count)
        let n = vDSP_Length(samples.count)
        // vDSP_vsmul: out = samples * gain (single SIMD pass).
        // vDSP_vclip: in-place clip to [-1, 1]. Replaces the `samples.map
        // { min(max(s * gain, -1), 1) }` per-element loop.
        vDSP_vsmul(samples, 1, &gainLinear, &out, 1, n)
        vDSP_vclip(out, 1, &lowerBound, &upperBound, &out, 1, n)
        return (out, true)
    }

    private static func rmsDBFS(_ samples: [Float]) -> Double {
        // vDSP_svesq computes sum-of-squares in a single SIMD pass, replacing
        // the per-element `for s in samples { sumSquares += s² }` loop. Keeps
        // the Float accumulator — for normalized [-1, 1] PCM, even multi-minute
        // clips stay well within Float precision for the dBFS values we read
        // back here (we only care about ~1 dB resolution against a 12 dB band).
        var sumSquares: Float = 0
        vDSP_svesq(samples, 1, &sumSquares, vDSP_Length(samples.count))
        let meanSquare = Double(sumSquares) / Double(samples.count)
        guard meanSquare > 0 else { return -.infinity }
        let rms = sqrt(meanSquare)
        return 20.0 * log10(rms)
    }

    // MARK: - Streaming

    private static func chunkedStream(samples: [Float]) -> AsyncStream<PCMBufferChunk> {
        AsyncStream { continuation in
            var offset = 0
            while offset < samples.count {
                let end = min(offset + chunkSamples, samples.count)
                let chunk = PCMBufferChunk(
                    samples: Array(samples[offset..<end]),
                    startSampleIndex: offset
                )
                continuation.yield(chunk)
                offset = end
            }
            continuation.finish()
        }
    }
}

public enum PreprocessorError: Error, LocalizedError {
    case converterInitFailed
    case outputBufferAllocFailed
    case conversionFailed(underlying: String)

    public var errorDescription: String? {
        switch self {
        case .converterInitFailed:
            return "Failed to create AVAudioConverter between source and target formats"
        case .outputBufferAllocFailed:
            return "Failed to allocate output PCM buffer for resample"
        case .conversionFailed(let underlying):
            return "Audio conversion failed: \(underlying)"
        }
    }
}
