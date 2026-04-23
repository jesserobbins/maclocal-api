import Foundation
import XCTest

@testable import MacLocalAPI

final class SpeechTranscriberPoolTests: XCTestCase {

    func testWarmLocalePrecachedAtInit() async {
        let pool = await SpeechTranscriberPool(warmLocales: [Locale(identifier: "en-US")])
        let cached = await pool.isCached(
            locale: Locale(identifier: "en-US"),
            featureSet: .init(wantWordTimings: true)
        )
        XCTAssertTrue(cached)
    }

    func testCheckoutReturnsSameEngineForSameKey() async {
        let pool = await SpeechTranscriberPool(warmLocales: [])
        let e1 = await pool.checkout(locale: Locale(identifier: "en-US"))
        let e2 = await pool.checkout(locale: Locale(identifier: "en-US"))
        XCTAssertTrue(e1 === e2)
    }

    func testCheckoutReturnsDistinctEnginesForDifferentLocales() async {
        let pool = await SpeechTranscriberPool(warmLocales: [])
        let enEngine = await pool.checkout(locale: Locale(identifier: "en-US"))
        let esEngine = await pool.checkout(locale: Locale(identifier: "es-ES"))
        XCTAssertFalse(enEngine === esEngine)
    }

    func testFeatureSetIsPartOfKey() async {
        let pool = await SpeechTranscriberPool(warmLocales: [])
        let withTimings = await pool.checkout(
            locale: Locale(identifier: "en-US"),
            featureSet: .init(wantWordTimings: true)
        )
        let withoutTimings = await pool.checkout(
            locale: Locale(identifier: "en-US"),
            featureSet: .init(wantWordTimings: false)
        )
        XCTAssertFalse(withTimings === withoutTimings)
    }

    func testLRUEvictsNonPinnedWhenOverMaxSize() async {
        let pool = await SpeechTranscriberPool(
            warmLocales: [Locale(identifier: "en-US")],
            maxSize: 2
        )
        let fs = SpeechTranscriberPool.FeatureSet(wantWordTimings: true)
        // en-US is already there (pinned). Add es-ES, then fr-FR — that's 3
        // entries total. Pool's max=2, pinned en-US must remain, so es-ES
        // gets evicted (oldest non-pinned).
        _ = await pool.checkout(locale: Locale(identifier: "es-ES"), featureSet: fs)
        _ = await pool.checkout(locale: Locale(identifier: "fr-FR"), featureSet: fs)

        let esStillCached = await pool.isCached(locale: Locale(identifier: "es-ES"), featureSet: fs)
        let frCached = await pool.isCached(locale: Locale(identifier: "fr-FR"), featureSet: fs)
        let enCached = await pool.isCached(locale: Locale(identifier: "en-US"), featureSet: fs)

        XCTAssertFalse(esStillCached, "es-ES should have been evicted when fr-FR checked out")
        XCTAssertTrue(frCached, "fr-FR should be cached")
        XCTAssertTrue(enCached, "en-US is pinned and must not be evicted")
    }

    func testTouchUpdatesLRUOrder() async {
        let pool = await SpeechTranscriberPool(warmLocales: [], maxSize: 2)
        let fs = SpeechTranscriberPool.FeatureSet(wantWordTimings: true)
        _ = await pool.checkout(locale: Locale(identifier: "es-ES"), featureSet: fs)
        _ = await pool.checkout(locale: Locale(identifier: "fr-FR"), featureSet: fs)
        // Touch es-ES so it becomes most-recent, then add a third key.
        _ = await pool.checkout(locale: Locale(identifier: "es-ES"), featureSet: fs)
        _ = await pool.checkout(locale: Locale(identifier: "de-DE"), featureSet: fs)

        let esCached = await pool.isCached(locale: Locale(identifier: "es-ES"), featureSet: fs)
        let frCached = await pool.isCached(locale: Locale(identifier: "fr-FR"), featureSet: fs)
        let deCached = await pool.isCached(locale: Locale(identifier: "de-DE"), featureSet: fs)

        XCTAssertTrue(esCached, "es-ES touched recently; should survive")
        XCTAssertFalse(frCached, "fr-FR was least-recently-used; should be evicted")
        XCTAssertTrue(deCached, "de-DE newly checked out")
    }
}
