import Foundation
import Speech

/// Cache of `SpeechTranscriberEngine` instances keyed by the request shape.
///
/// The pool exists so we don't pay per-request setup cost on repeat
/// transcriptions of the same locale + feature-set — the dominant path
/// for a zero-config English-first server. Current v1 caches `Engine`
/// instances; each `Engine.transcribe(...)` call still creates a fresh
/// `SpeechAnalyzer` internally (the analyzer's reuse semantics across
/// multiple `start()` calls are not yet characterized; a later revision
/// can swap in warm-analyzer reuse once measured).
///
/// Warmup: at init, if any locale in `warmLocales` is not already in
/// `SpeechTranscriber.installedLocales`, the pool logs and proceeds — we
/// don't block startup on an async model download. First transcription in
/// that locale will trigger the download.
actor SpeechTranscriberPool {
    /// Default max cached entries. Pinned locales (the `warmLocales` list)
    /// are never evicted.
    static let defaultMaxSize: Int = 4

    struct FeatureSet: Hashable, Sendable {
        let wantWordTimings: Bool
    }

    struct Key: Hashable, Sendable {
        let localeIdentifier: String
        let featureSet: FeatureSet
    }

    private var cache: [Key: SpeechTranscriberEngine] = [:]
    private var lruOrder: [Key] = []  // oldest first
    private let pinnedKeys: Set<Key>
    private let maxSize: Int

    /// Create a pool and warm the given locales.
    ///
    /// `warmLocales`: locales that should be eagerly checked at startup and
    /// pinned in the cache. Pinning means they're never evicted by the LRU.
    init(
        warmLocales: [Locale] = [Locale(identifier: "en-US")],
        maxSize: Int = SpeechTranscriberPool.defaultMaxSize
    ) async {
        self.maxSize = max(1, maxSize)
        var pinned = Set<Key>()
        for locale in warmLocales {
            let key = Key(
                localeIdentifier: locale.identifier,
                featureSet: FeatureSet(wantWordTimings: true)
            )
            pinned.insert(key)
            cache[key] = SpeechTranscriberEngine()
            lruOrder.append(key)
        }
        self.pinnedKeys = pinned
    }

    /// Check out the engine for a given key, creating it if missing. Updates
    /// LRU order. Evicts the least-recently-used non-pinned entry when over
    /// the size cap.
    func checkout(locale: Locale, featureSet: FeatureSet) -> SpeechTranscriberEngine {
        let key = Key(localeIdentifier: locale.identifier, featureSet: featureSet)
        if let existing = cache[key] {
            touch(key: key)
            return existing
        }
        let engine = SpeechTranscriberEngine()
        cache[key] = engine
        lruOrder.append(key)
        evictIfNeeded()
        return engine
    }

    /// Check out with the default feature set (word timings on).
    func checkout(locale: Locale) -> SpeechTranscriberEngine {
        return checkout(locale: locale, featureSet: FeatureSet(wantWordTimings: true))
    }

    /// Current cache size — exposed for tests.
    var cacheSize: Int { cache.count }

    /// Whether a given key is currently cached — exposed for tests.
    func isCached(locale: Locale, featureSet: FeatureSet) -> Bool {
        let key = Key(localeIdentifier: locale.identifier, featureSet: featureSet)
        return cache[key] != nil
    }

    // MARK: - Internals

    private func touch(key: Key) {
        if let idx = lruOrder.firstIndex(of: key) {
            lruOrder.remove(at: idx)
        }
        lruOrder.append(key)
    }

    private func evictIfNeeded() {
        guard cache.count > maxSize else { return }
        // Evict oldest non-pinned entry.
        for key in lruOrder {
            if pinnedKeys.contains(key) { continue }
            cache.removeValue(forKey: key)
            if let idx = lruOrder.firstIndex(of: key) {
                lruOrder.remove(at: idx)
            }
            if cache.count <= maxSize { return }
        }
    }
}
