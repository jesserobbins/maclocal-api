# Speech API Spike Findings — 2026-04-23

**Method:** Static analysis of the Speech framework `.swiftinterface` in the installed macOS 26.4 SDK:
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.4.sdk/System/Library/Frameworks/Speech.framework/Modules/Speech.swiftmodule/arm64e-apple-macos.swiftinterface`

Runtime measurement deferred — a full runtime probe requires Speech Recognition TCC consent and a live audio handle, which are not reachable from this headless Claude Code session. The static analysis below resolves the primary design uncertainty (the API shape). Two secondary measurements (per-analyzer memory cost, streaming cancel-and-restart) are downgraded to observations to make during implementation of Tasks 8 (pool) and 9 (reassessor), where they fall out naturally of the test fixtures we'll be writing anyway.

## 1. AnalysisContext.contextualStrings — **CONFIRMED**

```swift
final public class AnalysisContext : Swift.Sendable {
    final public var contextualStrings: [AnalysisContext.ContextualStringsTag : [String]]
    final public var userData: [AnalysisContext.UserDataTag : any Sendable]
    // ...
}

extension AnalysisContext.ContextualStringsTag {
    public static let general: AnalysisContext.ContextualStringsTag
    // rawValue-based init exists; additional tags may be defined elsewhere in the framework
}
```

- `contextualStrings` is a `[Tag : [String]]` dictionary — runtime plaintext, keyed by tag.
- `.general` tag is confirmed public; additional tags may exist (the type has a public `init(rawValue:)` so it's extensible).
- The spec's Section 3 assumption — "plaintext strings at runtime, no precompile needed" — is correct. We bundle plaintext `en.txt`, load at startup, and feed under `.general`.

## 2. SpeechAnalyzer and SpeechTranscriber — **CONFIRMED**

```swift
final public actor SpeechAnalyzer : Sendable {
    public convenience init(modules: [any SpeechModule], options: Options? = nil)
    public convenience init<InputSequence>(
        inputSequence: InputSequence,
        modules: [any SpeechModule],
        options: Options? = nil,
        analysisContext: AnalysisContext = .init(),
        volatileRangeChangedHandler: (...)? = nil
    ) where InputSequence: Sendable, InputSequence: AsyncSequence, InputSequence.Element == AnalyzerInput

    public convenience init(
        inputAudioFile: AVAudioFile,
        modules: [any SpeechModule],
        options: Options? = nil,
        analysisContext: AnalysisContext = .init(),
        finishAfterFile: Bool = false,
        volatileRangeChangedHandler: (...)? = nil
    ) async throws

    final public var context: AnalysisContext { get }
    final public func setContext(_ newContext: AnalysisContext) async throws
}

final public class SpeechTranscriber : SpeechModule, LocaleDependentSpeechModule {
    convenience public init(locale: Locale, preset: Preset)
    final public var results: some AsyncSequence<Result, Error>

    // Presets:
    // .transcription, .transcriptionWithAlternatives,
    // .timeIndexedTranscriptionWithAlternatives,
    // .progressiveTranscription, .timeIndexedProgressiveTranscription
}
```

Implications for our design:

- Construction with `inputAudioFile:` is the non-streaming path for short benchmark cases.
- Construction with `inputSequence:` is the streaming path we need for overlapping preprocess + inference, and for implementing the cancel-and-restart in `LanguageReassessor`.
- `analysisContext` is a constructor argument AND there's a `setContext(_:)` method — context can be updated mid-session.
- `SpeechAnalyzer.Options` carries `priority: TaskPriority` and `modelRetention: ModelRetention` — the retention hint is directly relevant to our pool design (Task 8).
- Preset `.timeIndexedProgressiveTranscription` gives us word timings + streaming in one preset — exactly what the spec wants.

## 3. Result shape (word timings, confidence, alternatives) — **CONFIRMED**

```swift
public struct SpeechTranscriber.Result : SpeechModuleResult, Sendable, ... {
    public let range: CMTimeRange                 // audio time range of this result
    public let resultsFinalizationTime: CMTime
    public var text: AttributedString             // per-run attributes include audioTimeRange + confidence
    public let alternatives: [AttributedString]
    // isFinal surfaced via a SpeechModuleResult conformance
}

extension AttributedString {
    public func rangeOfAudioTimeRangeAttributes(intersecting timeRange: CMTimeRange) -> Range<Index>?
}

// Attribute scope exposes:
//   .audioTimeRange      -- per-word CMTimeRange
//   .confidence          -- per-word confidence score
```

- Word-level timings are per-character (or per-word) attributes on the `AttributedString`. Implementation pattern: iterate over `text.runs` and extract `audioTimeRange` + `confidence`.
- `alternatives` is populated only when the transcriber's preset includes `.transcriptionWithAlternatives` or `.timeIndexedTranscriptionWithAlternatives`. For zero-config callers we use the progressive preset (no alternatives overhead); for future alternatives-reranking work (out of scope for this spec), we'd switch presets.
- `isFinal` distinguishes volatile (in-flight) vs final results in the streaming mode.

## 4. DictationTranscriber — **alternative stronger-bias path (NOT the default)**

```swift
final public class DictationTranscriber : SpeechModule, LocaleDependentSpeechModule {
    public static func customizedLanguage(
        modelConfiguration: SFSpeechLanguageModel.Configuration
    ) -> ContentHint
    // Presets: .phrase, .shortDictation, .progressiveShortDictation,
    //          .longDictation, .progressiveLongDictation, .timeIndexedLongDictation
}
```

- `DictationTranscriber` accepts a `ContentHint.customizedLanguage` fed from `SFSpeechLanguageModel.Configuration` — the *compiled* custom LM path.
- `SpeechTranscriber` does **not** have this content hint; it uses only `AnalysisContext.contextualStrings`.
- DictationTranscriber output is "dictation style" (punctuation-aware) vs SpeechTranscriber's "raw words."

**Design decision:** default engine stays `SpeechTranscriber` (raw words, matches whisper output style and OpenAI wire shape). The "spec Section 7 spike-fail fallback" — wiring DictationTranscriber + SFSpeechLanguageModel for tech-domain cases when Gate B demands stronger bias — is confirmed to be a real, API-supported path and NO LONGER requires falling back to the deprecated `SFSpeechRecognizer`. It's a cleaner fallback than the spec anticipated.

## 5. AssetInventory — **CONFIRMED (locale model download path)**

```swift
final public class AssetInventory {
    public static func status(forModules modules: [any SpeechModule]) async -> AssetInventory.Status
}

// On the transcriber types:
public static var supportedLocales: [Foundation.Locale] { get async }
public static var installedLocales: [Foundation.Locale] { get }
```

- First use of a locale may trigger a model download (user-visible, probably via a system asset-installation activity).
- `SpeechTranscriberPool` warmup at startup needs to reckon with this: for `en-US` cold-start we may pay a one-time download cost. Pool init should call `installedLocales` and, if `en-US` is absent, initiate download before the server accepts requests, or fall back gracefully.
- This is a small addition to Task 8 (pool) that wasn't in the spec but is necessary for correct zero-config behavior.

## 6. Authorization — **unchanged path**

Static analysis did not surface a new authorization API specific to `SpeechAnalyzer`; the existing `SFSpeechRecognizer.authorizationStatus()` / `SFSpeechRecognizer.requestAuthorization(_:)` path still applies (and the current `Models/SpeechService.swift` already handles it correctly). The `promptForAuthorization` flag and headless-refuses-TCC behavior carry over unchanged.

## 7. Deferred measurements

| Measurement | Why deferred | Where picked up |
|---|---|---|
| Per-analyzer memory cost | Needs running analyzers with loaded models | Task 8 — the pool's sizing test will measure this naturally, and the size cap becomes a runtime configuration after calibration |
| Streaming cancel-and-restart semantics | Needs a live streaming session | Task 9 — the reassessor's integration test will exercise cancel-and-restart against `spanish-speech.wav`; if it misbehaves, the test fails loudly |

These deferrals do not block Tasks 3–7 from proceeding.

## Decision

- [x] **Proceed with spec as written**, with these small clarifications:
  1. `AnalysisContext.contextualStrings` tag dictionary — use `.general` tag for all bundled/env/project/request vocab in v1.
  2. `SpeechAnalyzer.Options.modelRetention` is a real hint — set to favor retention on the pinned `en-US` pool slot and default for others.
  3. Pool init must check `SpeechTranscriber.installedLocales` and trigger download via `AssetInventory` for the pinned locale if missing, or surface a clear "model not installed" error.
  4. Task 9's "cancel-and-restart" mechanism relies on `SpeechAnalyzer` being safely releasable mid-stream. If the integration test shows this leaks or hangs, fall back to "wait-and-retry" (spec Section 5's alternative) and note the regression in the findings.
  5. DictationTranscriber + `SFSpeechLanguageModel.customizedLanguage` is a confirmed (not speculative) fallback for Gate B if `contextualStrings` turns out weaker than needed on the tech-domain subset — adds ~1 day to implementation, narrowly scoped to the tech-domain route.

- [ ] Proceed with spec, Gate B becomes report-only
- [ ] Add SFSpeechRecognizer + SFSpeechLanguageModel as second engine
- [ ] Escalate to human
