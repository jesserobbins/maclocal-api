import AVFoundation
import CoreMedia
import Foundation
import Speech

/// File-based wrapper around `SpeechAnalyzer` + `SpeechTranscriber`.
///
/// Single entry: `transcribe(url:locale:contextualStrings:wantWordTimings:)`
/// returns a `TranscriptionAttempt` containing the transcribed text plus the
/// signal values `LanguageReassessor` needs to decide on a retry:
/// mean early-window confidence, OOV ratio, and an n-gram locale guess.
///
/// Streaming input (for cancel-and-restart in the reassessor path) is left
/// for `SpeechTranscriberPool` + `SpeechService` in later tasks.
actor SpeechTranscriberEngine {
    /// Duration of the window at the start of audio over which we compute
    /// `meanEarlyConfidence`. Matches `LanguageReassessor.evaluationWindowSec`
    /// contract (3 s).
    static let earlyConfidenceWindowSec: Double = 3.0

    init() {}

    func transcribe(
        url: URL,
        locale: Locale,
        contextualStrings: [String],
        wantWordTimings: Bool = true
    ) async throws -> TranscriptionAttempt {
        let audioFile = try AVAudioFile(forReading: url)
        let sampleRate = audioFile.processingFormat.sampleRate
        let durationSec = sampleRate > 0 ? Double(audioFile.length) / sampleRate : 0.0

        // Choose a preset that exposes time + confidence attributes.
        let preset: SpeechTranscriber.Preset = wantWordTimings
            ? .timeIndexedProgressiveTranscription
            : .progressiveTranscription
        let transcriber = SpeechTranscriber(locale: locale, preset: preset)

        // Bias via AnalysisContext.contextualStrings under the general tag.
        let context = AnalysisContext()
        context.contextualStrings = [.general: contextualStrings]

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        try await analyzer.setContext(context)

        // Consumer runs concurrently with start(). When the analyzer reaches
        // end-of-file with finishAfterFile:true, the results sequence
        // terminates naturally.
        let resultsTask = Task<[SpeechTranscriber.Result], Error> {
            var collected: [SpeechTranscriber.Result] = []
            for try await r in transcriber.results {
                collected.append(r)
            }
            return collected
        }

        try await analyzer.start(inputAudioFile: audioFile, finishAfterFile: true)
        let results = try await resultsTask.value

        return Self.assembleAttempt(
            from: results,
            durationSec: durationSec,
            activeVocab: Set(contextualStrings.map { $0.lowercased() })
        )
    }

    // MARK: - Result assembly

    static func assembleAttempt(
        from results: [SpeechTranscriber.Result],
        durationSec: Double,
        activeVocab: Set<String>
    ) -> TranscriptionAttempt {
        var texts: [String] = []
        var words: [WordTiming] = []
        var segments: [Segment] = []
        var earlyConfidences: [Double] = []

        let earlyCutoff = CMTime(seconds: Self.earlyConfidenceWindowSec, preferredTimescale: 1_000)

        for result in results {
            let resultText = String(result.text.characters)
            texts.append(resultText)

            // Segment timing from result.range.
            let segStart = result.range.start
            let segEnd = CMTimeAdd(result.range.start, result.range.duration)
            let segConfs = extractConfidences(from: result.text)
            let segMeanConf = segConfs.isEmpty ? 0.0 : segConfs.reduce(0, +) / Double(segConfs.count)
            segments.append(Segment(
                text: resultText,
                startMs: cmTimeMs(segStart),
                endMs: cmTimeMs(segEnd),
                meanConfidence: segMeanConf
            ))

            // Per-run word timings + confidence.
            for run in result.text.runs {
                let runText = String(result.text[run.range].characters).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !runText.isEmpty else { continue }
                let timeRange: CMTimeRange? = run.audioTimeRange
                let confidence: Double = run.transcriptionConfidence ?? 0.0

                if let tr = timeRange {
                    words.append(WordTiming(
                        word: runText,
                        startMs: cmTimeMs(tr.start),
                        endMs: cmTimeMs(CMTimeAdd(tr.start, tr.duration)),
                        confidence: confidence
                    ))
                    // Pick up confidences from runs whose start falls in the
                    // early window.
                    if CMTimeCompare(tr.start, earlyCutoff) <= 0 {
                        earlyConfidences.append(confidence)
                    }
                } else {
                    // No time range available — still contribute to confidence
                    // signal if we consider the whole first N results "early".
                    earlyConfidences.append(confidence)
                }
            }
        }

        let fullText = texts.joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let meanEarly = earlyConfidences.isEmpty
            ? 1.0  // no signal → treat as high confidence so retry doesn't fire
            : earlyConfidences.reduce(0, +) / Double(earlyConfidences.count)

        let tokens = OOVCalculator.tokenize(fullText)
        let oov = OOVCalculator.oovRatio(tokens: tokens, known: activeVocab)

        let guess = CharacterClassLanguageID.identify(text: fullText)

        return TranscriptionAttempt(
            text: fullText,
            meanEarlyConfidence: meanEarly,
            oovRatio: oov,
            segments: segments,
            words: words,
            detectedLanguageGuess: guess
        )
    }

    // MARK: - Helpers

    static func extractConfidences(from attributed: AttributedString) -> [Double] {
        var out: [Double] = []
        for run in attributed.runs {
            if let c: Double = run.transcriptionConfidence {
                out.append(c)
            }
        }
        return out
    }

    static func cmTimeMs(_ time: CMTime) -> Int {
        guard time.isValid else { return 0 }
        return Int((CMTimeGetSeconds(time) * 1000.0).rounded())
    }
}
