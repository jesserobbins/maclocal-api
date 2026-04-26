import Foundation

/// Shared types for the macOS 26 Speech pipeline.
///
/// This module is the new surface that replaces the `SFSpeechRecognizer`-based
/// `Models/SpeechService.swift` implementation. During the incremental migration
/// the legacy `SpeechRequestOptions` and `SpeechError` continue to live in
/// `Models/SpeechService.swift`; they move here when the legacy service is
/// deleted in the final wiring task.

/// OpenAI-compatible response format for `POST /v1/audio/transcriptions`.
public enum SpeechResponseFormat: String, Sendable, Codable {
    case text
    case json
    case verboseJson = "verbose_json"
    case srt
    case vtt
}

/// Requested timestamp granularity for `verbose_json` responses.
public enum TimestampGranularity: String, Sendable, Codable {
    case word
    case segment
}

/// Per-word timing emitted by `SpeechTranscriber`. Timings are relative to the
/// start of the audio file; millisecond resolution is sufficient for the
/// OpenAI wire format.
public struct WordTiming: Sendable, Codable, Equatable {
    public let word: String
    public let startMs: Int
    public let endMs: Int
    public let confidence: Double

    public init(word: String, startMs: Int, endMs: Int, confidence: Double) {
        self.word = word
        self.startMs = startMs
        self.endMs = endMs
        self.confidence = confidence
    }
}

/// A segment returned in `verbose_json`. Segments are coarser than words and
/// usually correspond to a single `SpeechTranscriber.Result`.
public struct Segment: Sendable, Codable, Equatable {
    public let text: String
    public let startMs: Int
    public let endMs: Int
    public let meanConfidence: Double

    public init(text: String, startMs: Int, endMs: Int, meanConfidence: Double) {
        self.text = text
        self.startMs = startMs
        self.endMs = endMs
        self.meanConfidence = meanConfidence
    }
}

/// Intermediate result from a single transcription pass. Holds enough signal
/// for `LanguageReassessor` to decide whether to retry with a different locale
/// without needing the full `SpeechAnalyzer` output again.
public struct TranscriptionAttempt: Sendable {
    public let text: String
    /// Mean confidence over results falling in the first
    /// `LanguageReassessor.evaluationWindowSec` seconds of audio.
    public let meanEarlyConfidence: Double
    /// Ratio of emitted tokens that are out-of-vocabulary relative to the
    /// union of the active contextual vocab and the bundled English frequency
    /// list. Range [0.0, 1.0].
    public let oovRatio: Double
    public let segments: [Segment]
    public let words: [WordTiming]
    /// Locale guess from a lightweight character-n-gram identifier run against
    /// the first-pass text. `nil` when the identifier did not produce a
    /// confident guess.
    public let detectedLanguageGuess: Locale?

    public init(
        text: String,
        meanEarlyConfidence: Double,
        oovRatio: Double,
        segments: [Segment],
        words: [WordTiming],
        detectedLanguageGuess: Locale?
    ) {
        self.text = text
        self.meanEarlyConfidence = meanEarlyConfidence
        self.oovRatio = oovRatio
        self.segments = segments
        self.words = words
        self.detectedLanguageGuess = detectedLanguageGuess
    }
}

/// Final transcription result returned by `SpeechService` and serialized into
/// the HTTP response. Shape is a superset of the current plain-JSON contract;
/// the `json` and `text` response formats ignore the optional fields.
public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String
    public let durationMs: Int
    public let segments: [Segment]?
    public let words: [WordTiming]?
    public let languageReassessed: Bool

    public init(
        text: String,
        language: String,
        durationMs: Int,
        segments: [Segment]? = nil,
        words: [WordTiming]? = nil,
        languageReassessed: Bool = false
    ) {
        self.text = text
        self.language = language
        self.durationMs = durationMs
        self.segments = segments
        self.words = words
        self.languageReassessed = languageReassessed
    }
}

/// Output of `AudioPreprocessor.prepare(...)`. The consumer drives the
/// `SpeechAnalyzer` from `samples` (in-memory) or `stream` (chunked async),
/// depending on whether streaming or buffered consumption is preferred.
/// Metadata fields report what the preprocessor did for observability and
/// test assertions.
public struct PreparedAudio: Sendable {
    /// All preprocessed PCM samples in 16 kHz mono float32. The pipeline
    /// uses these directly to build the AnalyzerInput sequence so VAD-
    /// trimmed silence never reaches inference.
    public let samples: [Float]
    /// Async stream of PCM buffers at 16 kHz mono float32. Completes when the
    /// source audio is fully emitted. Convenience derivative of `samples`
    /// for callers that prefer streaming consumption.
    public let stream: AsyncStream<PCMBufferChunk>
    public let durationMs: Int
    public let sampleRate: Double
    public let wasResampled: Bool
    public let wasLoudnessNormalized: Bool
    public let wasSilenceTrimmed: Bool
    /// Total silence (ms) removed from leading + trailing. Zero when no trim
    /// happened. Used by callers to translate word/segment timings emitted by
    /// `SpeechAnalyzer` (which sees the trimmed audio) back to the original
    /// timeline by adding `leadingTrimMs` to every timestamp.
    public let silenceTrimmedMs: Int
    public let leadingTrimMs: Int

    public init(
        samples: [Float] = [],
        stream: AsyncStream<PCMBufferChunk>,
        durationMs: Int,
        sampleRate: Double,
        wasResampled: Bool,
        wasLoudnessNormalized: Bool,
        wasSilenceTrimmed: Bool = false,
        silenceTrimmedMs: Int = 0,
        leadingTrimMs: Int = 0
    ) {
        self.samples = samples
        self.stream = stream
        self.durationMs = durationMs
        self.sampleRate = sampleRate
        self.wasResampled = wasResampled
        self.wasLoudnessNormalized = wasLoudnessNormalized
        self.wasSilenceTrimmed = wasSilenceTrimmed
        self.silenceTrimmedMs = silenceTrimmedMs
        self.leadingTrimMs = leadingTrimMs
    }
}

/// A chunk of 16 kHz mono float32 samples produced by `AudioPreprocessor`.
/// Kept as a plain value type so it can cross actor boundaries without
/// touching the `AVAudioPCMBuffer` class from `@unchecked Sendable` wrappers.
public struct PCMBufferChunk: Sendable {
    public let samples: [Float]
    /// Offset of this chunk's first sample from the start of the source, in
    /// samples. Useful for coordinate translation when a chunk is reprocessed.
    public let startSampleIndex: Int

    public init(samples: [Float], startSampleIndex: Int) {
        self.samples = samples
        self.startSampleIndex = startSampleIndex
    }
}
