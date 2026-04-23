import XCTest

@testable import MacLocalAPI

final class ContextualVocabResolverTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() async throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ContextualVocabResolverTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Bundled

    func testBundledVocabLoadedAtInit() throws {
        let resolver = try ContextualVocabResolver(envFilePath: nil, projectFilePath: nil)
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        XCTAssertTrue(
            strings.contains(where: { $0.caseInsensitiveCompare("Kubernetes") == .orderedSame }),
            "Bundled vocab should include Kubernetes"
        )
        XCTAssertTrue(
            strings.contains(where: { $0.caseInsensitiveCompare("Anthropic") == .orderedSame }),
            "Bundled vocab should include Anthropic"
        )
    }

    func testBundledVocabSkipsCommentsAndBlanks() {
        let raw = "# comment\n\nKubernetes\n   \n# another\nAnthropic\n"
        let parsed = ContextualVocabResolver.parse(text: raw)
        XCTAssertEqual(parsed, ["Kubernetes", "Anthropic"])
    }

    // MARK: - Env file

    func testEnvFileMergesWithBundled() throws {
        let envFile = tempDir.appendingPathComponent("env-vocab.txt")
        try "tachyon\nflux capacitor\n".write(to: envFile, atomically: true, encoding: .utf8)

        let resolver = try ContextualVocabResolver(
            envFilePath: envFile.path,
            projectFilePath: nil
        )
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        XCTAssertTrue(strings.contains("tachyon"))
        XCTAssertTrue(strings.contains("flux capacitor"))
        // Bundled still present.
        XCTAssertTrue(strings.contains(where: { $0.caseInsensitiveCompare("Kubernetes") == .orderedSame }))
    }

    func testMissingEnvFileIsNoOp() throws {
        let resolver = try ContextualVocabResolver(
            envFilePath: "/tmp/definitely-does-not-exist-\(UUID().uuidString).txt",
            projectFilePath: nil
        )
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        XCTAssertFalse(strings.isEmpty, "Bundled vocab should still load when env file absent")
    }

    // MARK: - Project file

    func testProjectFileMerges() throws {
        let projectFile = tempDir.appendingPathComponent("project-vocab.txt")
        try "# project vocab\nZeppelin\n".write(to: projectFile, atomically: true, encoding: .utf8)

        let resolver = try ContextualVocabResolver(
            envFilePath: nil,
            projectFilePath: projectFile.path
        )
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        XCTAssertTrue(strings.contains("Zeppelin"))
    }

    // MARK: - Per-request prompt

    func testPerRequestPromptAppearsFirst() throws {
        let resolver = try ContextualVocabResolver(envFilePath: nil, projectFilePath: nil)
        let strings = resolver.resolve(prompt: "tachyon flux", locale: "en-US")
        // Full-prompt entry + tokens come ahead of bundled entries.
        XCTAssertEqual(strings.first, "tachyon flux")
        XCTAssertTrue(strings.contains("tachyon"))
        XCTAssertTrue(strings.contains("flux"))
    }

    func testDedupCaseFold() throws {
        let envFile = tempDir.appendingPathComponent("dup.txt")
        try "kubernetes\n".write(to: envFile, atomically: true, encoding: .utf8)

        let resolver = try ContextualVocabResolver(
            envFilePath: envFile.path,
            projectFilePath: nil
        )
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        let loweredMatches = strings.filter { $0.caseInsensitiveCompare("Kubernetes") == .orderedSame }
        XCTAssertEqual(loweredMatches.count, 1, "Case-insensitive duplicates should be deduped")
    }

    func testPrecedencePreservesFirstOccurrenceCasing() throws {
        let envFile = tempDir.appendingPathComponent("env.txt")
        try "KUBERNETES\n".write(to: envFile, atomically: true, encoding: .utf8)

        let resolver = try ContextualVocabResolver(
            envFilePath: envFile.path,
            projectFilePath: nil
        )
        // Env comes before bundled → env casing "KUBERNETES" should win.
        let strings = resolver.resolve(prompt: nil, locale: "en-US")
        let match = strings.first(where: { $0.caseInsensitiveCompare("Kubernetes") == .orderedSame })
        XCTAssertEqual(match, "KUBERNETES")
    }

    // MARK: - Tokenization edge cases

    func testTokenizeNilPromptIsEmpty() {
        XCTAssertEqual(ContextualVocabResolver.tokenize(prompt: nil), [])
    }

    func testTokenizeStripsPunctuation() {
        let toks = ContextualVocabResolver.tokenize(prompt: "Kubernetes, orchestrates: microservices.")
        // First entry is the full string; then individual words.
        XCTAssertEqual(toks.first, "Kubernetes, orchestrates: microservices.")
        XCTAssertTrue(toks.contains("Kubernetes"))
        XCTAssertTrue(toks.contains("orchestrates"))
        XCTAssertTrue(toks.contains("microservices"))
    }

    func testTokenizeDropsOneCharTokens() {
        let toks = ContextualVocabResolver.tokenize(prompt: "a b Kubernetes c")
        XCTAssertFalse(toks.contains("a"))
        XCTAssertFalse(toks.contains("b"))
        XCTAssertTrue(toks.contains("Kubernetes"))
    }
}
