# AFM - Apple Foundation Models API
# Makefile for building and distributing the portable CLI

.PHONY: build clean clean-reports clean-benchmarks clean-corpus clean-all install uninstall portable dist test test-vision help submodules submodule-status webui build-with-webui patch patch-check

PATCH_SH  := Scripts/apply-mlx-patches.sh
PATCH_STAMP := vendor/mlx-swift-lm/.patches-applied

# Default target
all: build

# Apply vendor patches (idempotent — stamp file tracks state)
$(PATCH_STAMP): $(PATCH_SH) $(wildcard Scripts/patches/*)
	@echo "🩹 Applying vendor patches..."
	@bash $(PATCH_SH)
	@touch $(PATCH_STAMP)

patch: $(PATCH_STAMP)

patch-check:
	@bash $(PATCH_SH) --check

# Build the release binary (portable by default)
build: $(PATCH_STAMP)
	@echo "🔨 Building AFM..."
	@swift build -c release \
		--product afm \
		-Xswiftc -O \
		-Xswiftc -whole-module-optimization \
		-Xswiftc -cross-module-optimization
	@strip .build/release/afm
	@echo "✅ Build complete: .build/release/afm"
	@echo "📊 Size: $$(ls -lh .build/release/afm | awk '{print $$5}')"

# Build with enhanced portability optimizations
portable:
	@./build-portable.sh

# Initialize git submodules (pinned to specific commit for reproducibility)
# NOTE: llama.cpp is pinned to a specific commit - do not use --remote flag
submodules:
	@echo "📦 Initializing git submodules (pinned version)..."
	@git submodule update --init
	@echo "✅ Submodules initialized (llama.cpp @ $$(cd vendor/llama.cpp && git rev-parse --short HEAD))"

# Show pinned submodule versions
submodule-status:
	@echo "📌 Pinned submodule versions:"
	@git submodule status

# Build the webui from llama.cpp
webui: submodules
	@echo "🌐 Building webui..."
	@if [ ! -d "vendor/llama.cpp/tools/server/webui" ]; then \
		echo "❌ Error: webui source not found. Run 'make submodules' first."; \
		exit 1; \
	fi
	@cd vendor/llama.cpp/tools/server/webui && npm install && npm run build
	@mkdir -p Resources/webui
	@cp vendor/llama.cpp/tools/server/public/index.html.gz Resources/webui/
	@echo "✅ WebUI built: Resources/webui/index.html.gz"

# Build with webui included
build-with-webui: webui build
	@echo "✅ Build with webui complete"

# Clean build artifacts and revert vendor patches
clean:
	@echo "🧹 Cleaning build artifacts..."
	@if [ -f $(PATCH_STAMP) ]; then bash $(PATCH_SH) --revert; rm -f $(PATCH_STAMP); fi
	@swift package clean
	@rm -rf .build
	@rm -f dist/*.tar.gz
	@echo "✅ Clean complete"

# Clean test report artifacts (assertion HTML/JSONL reports)
clean-reports:
	@echo "🧹 Cleaning test reports..."
	@rm -rf test-reports/assertions-report-*.html test-reports/assertions-report-*.jsonl
	@rm -rf test-reports/smart-analysis-*.md
	@rm -rf test-reports/prefix-cache-bench-*
	@echo "  $$(find test-reports -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ') files remaining in test-reports/"
	@echo "✅ Reports cleaned"

# Clean benchmark results (vision/speech JSONL + HTML reports)
clean-benchmarks:
	@echo "🧹 Cleaning benchmark results..."
	@rm -rf Scripts/benchmark-results/vision-speech-*
	@rm -rf Scripts/benchmark-results/concurrency-benchmark-*
	@echo "✅ Benchmarks cleaned"

# Clean generated test corpus (binary images/audio — ground truth .txt preserved)
clean-corpus:
	@echo "🧹 Cleaning generated test corpus (preserving ground truth .txt)..."
	@rm -f Scripts/test-data/vision/*.jpg Scripts/test-data/vision/*.png Scripts/test-data/vision/*.pdf
	@rm -f Scripts/test-data/speech/*.wav Scripts/test-data/speech/*.mp3 Scripts/test-data/speech/*.m4a
	@echo "✅ Corpus cleaned (run Scripts/generate-test-corpus.sh to regenerate)"

# Clean everything: build + reports + benchmarks
clean-all: clean clean-reports clean-benchmarks
	@echo "✅ All clean"

# Install to system (requires sudo)
install: build
	@echo "📦 Installing AFM to /usr/local/bin..."
	@sudo cp .build/release/afm /usr/local/bin/afm
	@sudo chmod +x /usr/local/bin/afm
	@echo "✅ AFM installed to /usr/local/bin/afm"

# Uninstall from system
uninstall:
	@echo "🗑️  Uninstalling AFM..."
	@sudo rm -f /usr/local/bin/afm
	@echo "✅ AFM uninstalled"

# Create distribution package
dist: portable
	@./create-distribution.sh

# Test the binary
test: build
	@echo "🧪 Testing AFM binary..."
	@./.build/release/afm --help > /dev/null && echo "✅ Binary test passed" || echo "❌ Binary test failed"
	@cp .build/release/afm /tmp/afm-test-$$$$ && \
		/tmp/afm-test-$$$$ --version > /dev/null 2>&1 && \
		echo "✅ Portability test passed" || echo "⚠️  Portability test failed"; \
		rm -f /tmp/afm-test-$$$$

# Run vision/speech benchmark suite (auto-starts server)
test-vision: build
	@./Scripts/test-vision-speech.sh

# Development build (debug)
debug: $(PATCH_STAMP)
	@echo "🐛 Building debug version..."
	@swift build
	@echo "✅ Debug build complete: .build/debug/afm"

# Run the server (development)
run: debug
	@echo "🚀 Starting AFM server..."
	@./.build/debug/afm --port 9999

# Show help
help:
	@echo "AFM - Apple Foundation Models API"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build           - Build release binary (default, patches+portable)"
	@echo "  portable        - Build with enhanced portability"
	@echo "  clean           - Clean build artifacts and revert patches"
	@echo "  patch           - Apply vendor patches only"
	@echo "  patch-check     - Verify vendor patch status"
	@echo "  install         - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall       - Remove from /usr/local/bin"
	@echo "  dist            - Create distribution package"
	@echo "  test            - Test the binary and portability"
	@echo "  test-vision     - Run vision/speech benchmarks (auto-starts server)"
	@echo "  debug           - Build debug version"
	@echo "  run             - Build and run debug server"
	@echo "  submodules      - Initialize git submodules"
	@echo "  webui           - Build webui from llama.cpp (requires Node.js)"
	@echo "  build-with-webui - Build with webui included"
	@echo "  clean-reports   - Remove test report HTML/JSONL files"
	@echo "  clean-benchmarks - Remove benchmark result files"
	@echo "  clean-corpus    - Remove generated test images/audio (keeps .txt ground truth)"
	@echo "  clean-all       - Clean build + reports + benchmarks"
	@echo "  help            - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make build              # Build portable executable"
	@echo "  make build-with-webui   # Build with webui support"
	@echo "  make install            # Build and install to system"
	@echo "  make test-vision        # Run vision/speech benchmarks"
	@echo "  make clean-all          # Clean everything"
	@echo "  make test               # Test binary works"
	@echo ""
	@echo "Output: .build/release/afm (portable executable)"
