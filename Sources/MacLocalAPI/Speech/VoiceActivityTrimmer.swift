import Accelerate
import Foundation

/// Trims leading and trailing silence from a 16 kHz mono float32 sample buffer.
///
/// We deliberately do not touch internal silences in this pass. Mid-clip
/// silence removal would require a segment map so downstream word timings can
/// be translated back to the original timeline; leading/trailing trim only
/// shifts every word timing by a single constant (`leadingTrimmedSamples`),
/// which is trivial to compensate for. Real-world audio (voicemails, phone
/// captures, hand-recorded clips) is dominated by leading/trailing silence
/// anyway, so this captures most of the latency win whisper.cpp gets from VAD
/// without the timing-coordinate complexity.
public enum VoiceActivityTrimmer {
    /// Window length in samples. 20 ms at 16 kHz = 320 samples — short enough
    /// to resolve onset/offset cleanly, long enough that per-window RMS isn't
    /// dominated by individual sample noise.
    public static let windowSamples: Int = 320

    /// Padding (samples) preserved on each side of the kept region. 100 ms at
    /// 16 kHz prevents cutting word starts/ends when a soft consonant sits in
    /// the window just below threshold.
    public static let hangoverSamples: Int = 1_600

    /// If the proposed trim would remove less than this many samples total,
    /// skip the trim entirely. Avoids paying any overhead for already-tight
    /// audio and avoids tiny adjustments that aren't worth the timing offset.
    public static let minimumTrimSamples: Int = 3_200 // 200 ms at 16 kHz

    /// Absolute floor: any window with RMS below this is *never* treated as
    /// activity, regardless of the relative threshold. Prevents accidentally
    /// declaring noise floor "active" on extremely quiet recordings.
    public static let silenceFloorDBFS: Float = -50.0

    /// Relative margin below the clip's peak window RMS. A window is active
    /// iff its RMS exceeds `peakRMS - relativeMarginDB`. Speech usually lives
    /// within 10–20 dB of peak; ambient noise is 30–50 dB below. -25 dB lands
    /// in the gap, robust to absolute signal level.
    public static let relativeMarginDB: Float = 25.0

    public struct TrimResult: Sendable {
        public let samples: [Float]
        public let leadingTrimmedSamples: Int
        public let trailingTrimmedSamples: Int
        public let didTrim: Bool

        public init(
            samples: [Float],
            leadingTrimmedSamples: Int,
            trailingTrimmedSamples: Int,
            didTrim: Bool
        ) {
            self.samples = samples
            self.leadingTrimmedSamples = leadingTrimmedSamples
            self.trailingTrimmedSamples = trailingTrimmedSamples
            self.didTrim = didTrim
        }

        public static let untouched = TrimResult(
            samples: [],
            leadingTrimmedSamples: 0,
            trailingTrimmedSamples: 0,
            didTrim: false
        )
    }

    public static func trim(_ samples: [Float]) -> TrimResult {
        guard samples.count >= windowSamples else {
            return TrimResult(
                samples: samples,
                leadingTrimmedSamples: 0,
                trailingTrimmedSamples: 0,
                didTrim: false
            )
        }

        let windowCount = samples.count / windowSamples
        let windowRMS = computeWindowRMS(samples: samples, windowCount: windowCount)

        // Peak window RMS in linear units. If the whole clip is below the
        // absolute silence floor, treat as untouchable — the caller's
        // transcriber will decide what to do with apparently-silent input.
        guard let peakRMS = windowRMS.max(), peakRMS > 0 else {
            return TrimResult(
                samples: samples,
                leadingTrimmedSamples: 0,
                trailingTrimmedSamples: 0,
                didTrim: false
            )
        }

        let absoluteFloor = dbToLinear(silenceFloorDBFS)
        let relativeFloor = peakRMS * dbToLinear(-relativeMarginDB)
        let threshold = max(absoluteFloor, relativeFloor)

        var firstActiveWindow = -1
        var lastActiveWindow = -1
        for i in 0..<windowCount {
            if windowRMS[i] > threshold {
                if firstActiveWindow < 0 { firstActiveWindow = i }
                lastActiveWindow = i
            }
        }

        guard firstActiveWindow >= 0 else {
            // No window crossed the threshold — keep the original samples.
            return TrimResult(
                samples: samples,
                leadingTrimmedSamples: 0,
                trailingTrimmedSamples: 0,
                didTrim: false
            )
        }

        let activeStart = firstActiveWindow * windowSamples
        let activeEnd = (lastActiveWindow + 1) * windowSamples // exclusive

        let paddedStart = max(0, activeStart - hangoverSamples)
        let paddedEnd = min(samples.count, activeEnd + hangoverSamples)

        let leadingTrim = paddedStart
        let trailingTrim = samples.count - paddedEnd

        if (leadingTrim + trailingTrim) < minimumTrimSamples {
            return TrimResult(
                samples: samples,
                leadingTrimmedSamples: 0,
                trailingTrimmedSamples: 0,
                didTrim: false
            )
        }

        let trimmed = Array(samples[paddedStart..<paddedEnd])
        return TrimResult(
            samples: trimmed,
            leadingTrimmedSamples: leadingTrim,
            trailingTrimmedSamples: trailingTrim,
            didTrim: true
        )
    }

    /// Per-window RMS computed via vDSP_rmsqv on each fixed-size slice. The
    /// trailing partial window (samples.count % windowSamples) is intentionally
    /// dropped: it's at most one window's worth of audio (20 ms) and including
    /// it would skew the peakRMS detection on very short tails.
    private static func computeWindowRMS(samples: [Float], windowCount: Int) -> [Float] {
        var output = [Float](repeating: 0, count: windowCount)
        let n = vDSP_Length(windowSamples)
        samples.withUnsafeBufferPointer { samplesPtr in
            guard let base = samplesPtr.baseAddress else { return }
            output.withUnsafeMutableBufferPointer { outPtr in
                guard let outBase = outPtr.baseAddress else { return }
                for i in 0..<windowCount {
                    vDSP_rmsqv(base.advanced(by: i * windowSamples), 1, outBase.advanced(by: i), n)
                }
            }
        }
        return output
    }

    private static func dbToLinear(_ db: Float) -> Float {
        return Foundation.pow(10.0, db / 20.0)
    }
}
