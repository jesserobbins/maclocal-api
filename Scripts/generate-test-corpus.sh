#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# generate-test-corpus.sh — Generates test corpus for vision/speech benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
#
# Produces test images (via HTML->sips), audio (via macOS say), and verifies
# that ground-truth .txt files exist for each generated asset.
#
# Usage:
#   ./Scripts/generate-test-corpus.sh [--force] [--vision-only] [--speech-only] [--verify]
#
# Idempotent: skips files that already exist unless --force is passed.
# Self-contained: works on any Mac with standard tools (sips, say, textutil).

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$SCRIPT_DIR/test-data/vision"
SPEECH_DIR="$SCRIPT_DIR/test-data/speech"
FORCE=false
VISION_ONLY=false
SPEECH_ONLY=false
VERIFY_ONLY=false

# Image dimensions
IMAGE_WIDTH=800
IMAGE_HEIGHT=1000
CARD_WIDTH=600
CARD_HEIGHT=350
CODE_WIDTH=900
CODE_HEIGHT=600
TABLE_WIDTH=850
TABLE_HEIGHT=400
MENU_WIDTH=700
MENU_HEIGHT=1000

# Audio settings
SPEECH_RATE=180  # words per minute for say command

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --force) FORCE=true; shift ;;
    --vision-only) VISION_ONLY=true; shift ;;
    --speech-only) SPEECH_ONLY=true; shift ;;
    --verify) VERIFY_ONLY=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "  ℹ️  $*"; }
ok()    { echo "  ✅ $*"; }
skip()  { echo "  ⏭️  $*  (exists)"; }
warn()  { echo "  ⚠️  $*"; }
fail()  { echo "  ❌ $*"; }

# Check if a command exists
has_cmd() { command -v "$1" &>/dev/null; }

# Generate image from HTML using qlmanage (Quick Look) rendering
# Usage: html_to_image "<html_content>" output.jpg width height
html_to_image() {
  local html_content="$1"
  local output_path="$2"
  local width="${3:-$IMAGE_WIDTH}"
  local height="${4:-$IMAGE_HEIGHT}"
  local tmp_html="/tmp/corpus-gen-$$.html"

  echo "$html_content" > "$tmp_html"

  # Use qlmanage (Quick Look) to render HTML to PNG thumbnail
  qlmanage -t -s "$width" -o /tmp "$tmp_html" >/dev/null 2>&1
  local ql_output="/tmp/$(basename "$tmp_html").png"

  if [[ -f "$ql_output" ]]; then
    if [[ "$output_path" == *.jpg || "$output_path" == *.jpeg ]]; then
      # Convert PNG to JPEG via sips
      sips -s format jpeg "$ql_output" --out "$output_path" >/dev/null 2>&1
      rm -f "$ql_output"
    else
      mv "$ql_output" "$output_path"
    fi
  fi

  rm -f "$tmp_html" 2>/dev/null || true

  # Verify output exists
  if [[ ! -f "$output_path" ]]; then
    warn "Could not generate $output_path"
    return 1
  fi
}

# Generate PDF with embedded text using Python
# Usage: text_to_pdf "text content" output.pdf [num_pages]
text_to_pdf() {
  local text_content="$1"
  local output_path="$2"
  local num_pages="${3:-1}"

  python3 - "$output_path" "$num_pages" <<PYTHON_EOF
import sys, re

output_path = sys.argv[1]
num_pages = int(sys.argv[2])

text = """$text_content"""

# Split text into pages by double-newline sections or page breaks
sections = [s.strip() for s in re.split(r'\n\n+', text) if s.strip()]

# Distribute sections across pages
pages_text = []
if num_pages > 1 and len(sections) > 1:
    per_page = max(1, len(sections) // num_pages)
    for i in range(0, len(sections), per_page):
        pages_text.append('\n'.join(sections[i:i+per_page]))
else:
    pages_text = [text]

# Ensure we have at least num_pages
while len(pages_text) < num_pages:
    pages_text.append("(continued)")

# Build PDF manually with text streams
objects = []
page_obj_ids = []

# Object 1: Catalog
objects.append(b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")

# Object 3: Font
objects.append(b"3 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n")

# Build page objects starting at obj 4
content_start = 4
page_refs = []
obj_id = content_start

for i, page_text in enumerate(pages_text):
    # Content stream
    lines = page_text.split('\n')
    pdf_ops = "BT /F1 11 Tf 72 740 Td 14 TL\n"
    for line in lines:
        # Escape PDF special chars
        safe = line.replace(chr(92), chr(92)+chr(92)).replace('(', chr(92)+'(').replace(')', chr(92)+')')
        pdf_ops += f"({safe}) '\n"
    pdf_ops += "ET"
    stream_bytes = pdf_ops.encode('latin-1', errors='replace')

    content_obj_id = obj_id
    page_obj_id = obj_id + 1

    objects.append(f"{content_obj_id} 0 obj<</Length {len(stream_bytes)}>>stream\n".encode() + stream_bytes + b"\nendstream\nendobj\n")
    objects.append(f"{page_obj_id} 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents {content_obj_id} 0 R/Resources<</Font<</F1 3 0 R>>>>>>endobj\n".encode())

    page_refs.append(f"{page_obj_id} 0 R")
    obj_id += 2

# Object 2: Pages
kids = " ".join(page_refs)
pages_obj = f"2 0 obj<</Type/Pages/Kids[{kids}]/Count {len(pages_text)}>>endobj\n".encode()

# Assemble PDF
pdf_bytes = b"%PDF-1.4\n"
offsets = {}

# Write catalog (obj 1)
offsets[1] = len(pdf_bytes)
pdf_bytes += objects[0]

# Write pages (obj 2)
offsets[2] = len(pdf_bytes)
pdf_bytes += pages_obj

# Write font (obj 3)
offsets[3] = len(pdf_bytes)
pdf_bytes += objects[1]

# Write content + page objects
for i, obj_data in enumerate(objects[2:]):
    real_id = content_start + i
    offsets[real_id] = len(pdf_bytes)
    pdf_bytes += obj_data

# Cross-reference table
xref_offset = len(pdf_bytes)
num_objs = max(offsets.keys()) + 1
pdf_bytes += f"xref\n0 {num_objs}\n".encode()
pdf_bytes += b"0000000000 65535 f \n"
for oid in range(1, num_objs):
    pdf_bytes += f"{offsets[oid]:010d} 00000 n \n".encode()

pdf_bytes += f"trailer<</Size {num_objs}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF".encode()

with open(output_path, 'wb') as f:
    f.write(pdf_bytes)
PYTHON_EOF

  if [[ -s "$output_path" ]]; then
    return 0
  fi
  warn "Could not generate PDF: $output_path"
  return 1
}

# Should we generate a file? (respects --force and existence check)
should_generate() {
  local path="$1"
  if [[ "$FORCE" == "true" ]]; then
    return 0
  fi
  if [[ -f "$path" ]]; then
    return 1
  fi
  return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# Vision Corpus Generation
# ═══════════════════════════════════════════════════════════════════════════════
generate_vision_corpus() {
  echo ""
  echo "═══ Generating Vision OCR Test Corpus ═══"
  echo ""
  mkdir -p "$VISION_DIR"

  # ─── receipt-grocery.jpg ──────────────────────────────────────────────────
  if should_generate "$VISION_DIR/receipt-grocery.jpg"; then
    info "Generating receipt-grocery.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Courier,monospace;font-size:14px;padding:40px;background:white;color:black;">
<pre style="line-height:1.6;">
FRESH MART GROCERY
123 Main Street
Anytown, CA 94301

Date: 2024-03-15  Time: 14:32

Organic Bananas      2.49
Whole Milk 1gal      4.99
Sourdough Bread      5.49
Chicken Breast lb    8.99
Baby Spinach 5oz     3.99
Greek Yogurt         1.79
Olive Oil 16oz       7.49
Brown Rice 2lb       3.29

Subtotal            38.52
Tax (8.5%)           3.27
TOTAL               41.79

VISA ****4821
Thank you for shopping!
</pre></body></html>
HTML
)" "$VISION_DIR/receipt-grocery.jpg" 500 700
    ok "receipt-grocery.jpg"
  else
    skip "receipt-grocery.jpg"
  fi

  # ─── receipt-restaurant.jpg ───────────────────────────────────────────────
  if should_generate "$VISION_DIR/receipt-restaurant.jpg"; then
    info "Generating receipt-restaurant.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Courier,monospace;font-size:14px;padding:40px;background:white;color:black;">
<pre style="line-height:1.6;">
THE GOLDEN FORK
456 Oak Avenue
Anytown, CA 94301

Server: Maria  Table: 12
Date: 2024-03-15  7:45 PM

Caesar Salad         12.95
Grilled Salmon       28.50
Mushroom Risotto     22.00
Glass Pinot Noir     14.00
Tiramisu              9.50

Subtotal             86.95
Tax (8.5%)            7.39
Total                94.34

Tip: ________
Total: ________

Thank you for dining with us!
</pre></body></html>
HTML
)" "$VISION_DIR/receipt-restaurant.jpg" 500 700
    ok "receipt-restaurant.jpg"
  else
    skip "receipt-restaurant.jpg"
  fi

  # ─── business-card.jpg ────────────────────────────────────────────────────
  if should_generate "$VISION_DIR/business-card.jpg"; then
    info "Generating business-card.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Helvetica,Arial,sans-serif;padding:40px;background:white;color:#222;">
<div style="border:2px solid #333;padding:30px;max-width:500px;">
<h2 style="margin:0;font-size:20px;">Sarah J. Mitchell</h2>
<p style="margin:5px 0;color:#555;font-size:14px;">Senior Vice President, Engineering</p>
<hr style="border:1px solid #ddd;margin:15px 0;">
<p style="margin:5px 0;font-size:16px;font-weight:bold;">TechVenture Inc.</p>
<p style="margin:3px 0;font-size:13px;">sarah.mitchell@techventure.io</p>
<p style="margin:3px 0;font-size:13px;">+1 (415) 555-0187</p>
<p style="margin:3px 0;font-size:13px;">www.techventure.io</p>
<p style="margin:10px 0 0;font-size:12px;color:#666;">555 Innovation Way, Suite 800<br>San Francisco, CA 94107</p>
</div>
</body></html>
HTML
)" "$VISION_DIR/business-card.jpg" "$CARD_WIDTH" "$CARD_HEIGHT"
    ok "business-card.jpg"
  else
    skip "business-card.jpg"
  fi

  # ─── screenshot-code.png ──────────────────────────────────────────────────
  if should_generate "$VISION_DIR/screenshot-code.png"; then
    info "Generating screenshot-code.png"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Menlo,Monaco,monospace;font-size:13px;padding:30px;background:#1e1e1e;color:#d4d4d4;">
<pre style="line-height:1.5;">
<span style="color:#c586c0;">import</span> <span style="color:#4ec9b0;">Foundation</span>

<span style="color:#569cd6;">struct</span> <span style="color:#4ec9b0;">WeatherService</span> {
    <span style="color:#569cd6;">let</span> apiKey: <span style="color:#4ec9b0;">String</span>
    <span style="color:#569cd6;">let</span> baseURL = <span style="color:#ce9178;">"https://api.weather.example.com/v2"</span>

    <span style="color:#569cd6;">func</span> <span style="color:#dcdcaa;">fetchForecast</span>(latitude: <span style="color:#4ec9b0;">Double</span>, longitude: <span style="color:#4ec9b0;">Double</span>) <span style="color:#569cd6;">async throws</span> -> <span style="color:#4ec9b0;">Forecast</span> {
        <span style="color:#569cd6;">let</span> url = <span style="color:#4ec9b0;">URL</span>(string: <span style="color:#ce9178;">"\(baseURL)/forecast?lat=\(latitude)&amp;lon=\(longitude)&amp;key=\(apiKey)"</span>)!
        <span style="color:#569cd6;">let</span> (data, response) = <span style="color:#c586c0;">try await</span> <span style="color:#4ec9b0;">URLSession</span>.shared.data(from: url)
        <span style="color:#c586c0;">guard let</span> httpResponse = response <span style="color:#569cd6;">as?</span> <span style="color:#4ec9b0;">HTTPURLResponse</span>,
              httpResponse.statusCode == <span style="color:#b5cea8;">200</span> <span style="color:#c586c0;">else</span> {
            <span style="color:#c586c0;">throw</span> <span style="color:#4ec9b0;">WeatherError</span>.networkFailure
        }
        <span style="color:#c586c0;">return try</span> <span style="color:#4ec9b0;">JSONDecoder</span>().decode(<span style="color:#4ec9b0;">Forecast</span>.self, from: data)
    }
}
</pre></body></html>
HTML
)" "$VISION_DIR/screenshot-code.png" "$CODE_WIDTH" "$CODE_HEIGHT"
    ok "screenshot-code.png"
  else
    skip "screenshot-code.png"
  fi

  # ─── table-financial.png ──────────────────────────────────────────────────
  if should_generate "$VISION_DIR/table-financial.png"; then
    info "Generating table-financial.png"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Helvetica,Arial,sans-serif;font-size:13px;padding:30px;background:white;color:black;">
<h3 style="margin-bottom:15px;">Quarterly Revenue Report ($ millions)</h3>
<table style="border-collapse:collapse;width:100%;">
<tr style="background:#f0f0f0;"><th style="border:1px solid #ccc;padding:8px;">Division</th><th style="border:1px solid #ccc;padding:8px;">Q1 2024</th><th style="border:1px solid #ccc;padding:8px;">Q2 2024</th><th style="border:1px solid #ccc;padding:8px;">Q3 2024</th><th style="border:1px solid #ccc;padding:8px;">Q4 2024</th><th style="border:1px solid #ccc;padding:8px;">YoY Growth</th></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Cloud Services</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">142.3</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">156.8</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">168.4</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">182.1</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">28.0%</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Enterprise Software</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">89.7</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">92.1</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">95.4</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">98.2</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">9.5%</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Hardware</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">67.2</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">58.9</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">71.3</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">84.6</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">25.9%</td></tr>
<tr><td style="border:1px solid #ccc;padding:8px;">Consulting</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">34.5</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">36.2</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">38.1</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">41.7</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">20.9%</td></tr>
<tr style="font-weight:bold;background:#f8f8f8;"><td style="border:1px solid #ccc;padding:8px;">Total</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">333.7</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">344.0</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">373.2</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">406.6</td><td style="border:1px solid #ccc;padding:8px;text-align:right;">21.8%</td></tr>
</table>
</body></html>
HTML
)" "$VISION_DIR/table-financial.png" "$TABLE_WIDTH" "$TABLE_HEIGHT"
    ok "table-financial.png"
  else
    skip "table-financial.png"
  fi

  # ─── multilang-french.jpg ─────────────────────────────────────────────────
  if should_generate "$VISION_DIR/multilang-french.jpg"; then
    info "Generating multilang-french.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Georgia,serif;font-size:16px;padding:50px;background:white;color:#222;line-height:1.8;">
<h2 style="margin-bottom:5px;">Les Misérables</h2>
<p style="color:#666;margin-bottom:20px;font-style:italic;">par Victor Hugo</p>
<p>Tant qu'il existera, par le fait des lois et des moeurs, une
damnation sociale créant artificiellement, en pleine civilisation,
des enfers, et compliquant d'une fatalité humaine la destinée qui
est divine; tant que les trois problèmes du siècle, la dégradation
de l'homme par le prolétariat, la déchéance de la femme par la
faim, l'atrophie de l'enfant par la nuit, ne seront pas résolus;
tant que l'asphyxie sociale sera possible; en d'autres termes, et
à un point de vue plus étendu encore, tant qu'il y aura sur la
terre ignorance et misère, des livres de la nature de celui-ci
pourront ne pas être inutiles.</p>
</body></html>
HTML
)" "$VISION_DIR/multilang-french.jpg" "$IMAGE_WIDTH" 600
    ok "multilang-french.jpg"
  else
    skip "multilang-french.jpg"
  fi

  # ─── multilang-japanese.jpg ───────────────────────────────────────────────
  if should_generate "$VISION_DIR/multilang-japanese.jpg"; then
    info "Generating multilang-japanese.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:'Hiragino Mincho ProN',serif;font-size:18px;padding:50px;background:white;color:#222;line-height:2.0;">
<h2 style="margin-bottom:5px;">吾輩は猫である</h2>
<p style="color:#666;margin-bottom:20px;">夏目漱石</p>
<p>吾輩は猫である。名前はまだ無い。</p>
<p>どこで生れたかとんと見当がつかぬ。</p>
<p>何でも薄暗いじめじめした所で</p>
<p>ニャーニャー泣いていた事だけは記憶している。</p>
<p>吾輩はここで始めて人間というものを見た。</p>
</body></html>
HTML
)" "$VISION_DIR/multilang-japanese.jpg" 600 500
    ok "multilang-japanese.jpg"
  else
    skip "multilang-japanese.jpg"
  fi

  # ─── book-page.jpg ───────────────────────────────────────────────────────
  if should_generate "$VISION_DIR/book-page.jpg"; then
    info "Generating book-page.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Georgia,serif;font-size:15px;padding:60px;background:#faf8f0;color:#222;line-height:1.8;">
<p>It is a truth universally acknowledged, that a single man in
possession of a good fortune, must be in want of a wife.</p>
<p>However little known the feelings or views of such a man may be
on his first entering a neighbourhood, this truth is so well
fixed in the minds of the surrounding families, that he is
considered as the rightful property of some one or other of
their daughters.</p>
</body></html>
HTML
)" "$VISION_DIR/book-page.jpg" "$IMAGE_WIDTH" 500
    ok "book-page.jpg"
  else
    skip "book-page.jpg"
  fi

  # ─── menu-restaurant.jpg ─────────────────────────────────────────────────
  if should_generate "$VISION_DIR/menu-restaurant.jpg"; then
    info "Generating menu-restaurant.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Georgia,serif;font-size:14px;padding:40px;background:#f9f5ed;color:#333;">
<h1 style="text-align:center;font-size:24px;margin-bottom:5px;">BELLA CUCINA</h1>
<p style="text-align:center;font-size:12px;color:#666;margin-bottom:20px;">Italian Restaurant &amp; Wine Bar</p>
<h3 style="border-bottom:1px solid #999;padding-bottom:5px;">ANTIPASTI</h3>
<p>Bruschetta al Pomodoro <span style="float:right;">12</span></p>
<p>Carpaccio di Manzo <span style="float:right;">16</span></p>
<p>Burrata con Prosciutto <span style="float:right;">18</span></p>
<p>Calamari Fritti <span style="float:right;">14</span></p>
<h3 style="border-bottom:1px solid #999;padding-bottom:5px;">PRIMI</h3>
<p>Spaghetti Carbonara <span style="float:right;">22</span></p>
<p>Risotto ai Funghi Porcini <span style="float:right;">24</span></p>
<p>Pappardelle al Ragu <span style="float:right;">23</span></p>
<p>Gnocchi alla Sorrentina <span style="float:right;">20</span></p>
<h3 style="border-bottom:1px solid #999;padding-bottom:5px;">SECONDI</h3>
<p>Branzino al Forno <span style="float:right;">32</span></p>
<p>Ossobuco alla Milanese <span style="float:right;">36</span></p>
<p>Pollo alla Parmigiana <span style="float:right;">26</span></p>
<p>Tagliata di Manzo <span style="float:right;">38</span></p>
<h3 style="border-bottom:1px solid #999;padding-bottom:5px;">DOLCI</h3>
<p>Tiramisu <span style="float:right;">12</span></p>
<p>Panna Cotta <span style="float:right;">10</span></p>
<p>Cannoli Siciliani <span style="float:right;">11</span></p>
<p>Affogato <span style="float:right;">9</span></p>
</body></html>
HTML
)" "$VISION_DIR/menu-restaurant.jpg" "$MENU_WIDTH" "$MENU_HEIGHT"
    ok "menu-restaurant.jpg"
  else
    skip "menu-restaurant.jpg"
  fi

  # ─── prescription-label.jpg ──────────────────────────────────────────────
  if should_generate "$VISION_DIR/prescription-label.jpg"; then
    info "Generating prescription-label.jpg"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Arial,sans-serif;font-size:11px;padding:20px;background:white;color:black;">
<div style="border:2px solid black;padding:15px;max-width:450px;">
<h3 style="margin:0;font-size:14px;">CENTRAL PHARMACY</h3>
<p style="margin:2px 0;font-size:10px;">1200 Health Blvd, Anytown CA 94301</p>
<p style="margin:2px 0;font-size:10px;">Phone: (555) 234-5678  DEA: BC1234567</p>
<hr style="margin:8px 0;">
<p><strong>Rx# 7834521</strong></p>
<p>Patient: DOE, JOHN M<br>DOB: 05/15/1980</p>
<p style="font-size:16px;font-weight:bold;margin:10px 0;">LISINOPRIL 10MG TABLETS</p>
<p style="font-size:13px;">Take one tablet by mouth daily</p>
<p>Qty: 30  Refills: 5</p>
<p>Dr. A. Williams  NPI: 1234567890</p>
<p>Filled: 03/15/2024  Discard after: 03/15/2025</p>
<div style="background:#ffeeee;border:1px solid #cc0000;padding:5px;margin-top:8px;font-size:10px;">
<strong>WARNING:</strong> May cause dizziness<br>
Do not take with potassium supplements
</div>
</div>
</body></html>
HTML
)" "$VISION_DIR/prescription-label.jpg" 500 450
    ok "prescription-label.jpg"
  else
    skip "prescription-label.jpg"
  fi

  # ─── rotated-scan.jpg ─────────────────────────────────────────────────────
  if should_generate "$VISION_DIR/rotated-scan.jpg"; then
    info "Generating rotated-scan.jpg (rotated receipt)"
    # Generate a simple receipt then rotate it 15 degrees to simulate a skewed scan
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Courier,monospace;font-size:14px;padding:40px;background:#f5f5f0;color:black;transform:rotate(3deg);margin-top:30px;">
<pre style="line-height:1.6;">
CORNER STORE
789 Elm Street

Date: 2024-06-22  09:15

Coffee Large         3.50
Muffin Blueberry     2.75
Newspaper            1.50
Gum Spearmint        1.25

Total                9.00
Cash                10.00
Change               1.00

Have a great day!
</pre></body></html>
HTML
)" "$VISION_DIR/rotated-scan.jpg" 450 500
    # Apply rotation via sips
    if has_cmd sips; then
      sips --rotate 15 "$VISION_DIR/rotated-scan.jpg" >/dev/null 2>&1 || true
    fi
    ok "rotated-scan.jpg"
  else
    skip "rotated-scan.jpg"
  fi

  # ─── low-quality-scan.jpg ───────────────────────────────────────────────────
  if should_generate "$VISION_DIR/low-quality-scan.jpg"; then
    info "Generating low-quality-scan.jpg (low-res degraded image)"
    html_to_image "$(cat <<'HTML'
<html><body style="font-family:Arial,sans-serif;font-size:16px;padding:40px;background:#f8f8f8;color:#333;">
<h3 style="margin-bottom:10px;">MEMO</h3>
<p><strong>To:</strong> All Staff</p>
<p><strong>From:</strong> Human Resources</p>
<p><strong>Date:</strong> March 1, 2024</p>
<p><strong>Subject:</strong> Office Closure Notice</p>
<hr>
<p>Please be advised that the office will be closed on Friday, March 15, 2024 for building maintenance. All employees should work from home on that day.</p>
<p>Contact HR at ext. 4500 with questions.</p>
</body></html>
HTML
)" "$VISION_DIR/low-quality-scan.jpg" 600 400
    # Degrade quality via sips (resize down then back up to simulate low-quality scan)
    if has_cmd sips; then
      sips --resampleWidth 200 "$VISION_DIR/low-quality-scan.jpg" >/dev/null 2>&1 || true
      sips --resampleWidth 600 "$VISION_DIR/low-quality-scan.jpg" >/dev/null 2>&1 || true
      sips -s formatOptions 20 "$VISION_DIR/low-quality-scan.jpg" --out "$VISION_DIR/low-quality-scan.jpg" >/dev/null 2>&1 || true
    fi
    ok "low-quality-scan.jpg"
  else
    skip "low-quality-scan.jpg"
  fi

  # ─── mixed-layout-newsletter.pdf ─────────────────────────────────────────
  if should_generate "$VISION_DIR/mixed-layout-newsletter.pdf"; then
    info "Generating mixed-layout-newsletter.pdf"
    text_to_pdf "$(cat "$VISION_DIR/mixed-layout-newsletter.txt")" "$VISION_DIR/mixed-layout-newsletter.pdf" 1
    ok "mixed-layout-newsletter.pdf"
  else
    skip "mixed-layout-newsletter.pdf"
  fi

  # ─── multipage-report.pdf ────────────────────────────────────────────────
  if should_generate "$VISION_DIR/multipage-report.pdf"; then
    info "Generating multipage-report.pdf"
    text_to_pdf "$(cat "$VISION_DIR/multipage-report.txt")" "$VISION_DIR/multipage-report.pdf" 5
    ok "multipage-report.pdf"
  else
    skip "multipage-report.pdf"
  fi

  # ─── invoice-standard.pdf ────────────────────────────────────────────────
  if should_generate "$VISION_DIR/invoice-standard.pdf"; then
    info "Generating invoice-standard.pdf"
    text_to_pdf "$(cat "$VISION_DIR/invoice-standard.txt")" "$VISION_DIR/invoice-standard.pdf" 1
    ok "invoice-standard.pdf"
  else
    skip "invoice-standard.pdf"
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Speech Corpus Generation
# ═══════════════════════════════════════════════════════════════════════════════
generate_speech_corpus() {
  echo ""
  echo "═══ Generating Speech Transcription Test Corpus ═══"
  echo ""
  mkdir -p "$SPEECH_DIR"

  # ─── short-5s.wav ─────────────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/short-5s.wav"; then
    info "Generating short-5s.wav (macOS say)"
    say -o "$SPEECH_DIR/short-5s.aiff" "Hello, this is a short test of the speech transcription system."
    # Convert AIFF to WAV via afconvert
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/short-5s.aiff" "$SPEECH_DIR/short-5s.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/short-5s.aiff"
    else
      # Rename as fallback (AIFF is still usable)
      mv "$SPEECH_DIR/short-5s.aiff" "$SPEECH_DIR/short-5s.wav"
    fi
    ok "short-5s.wav"
  else
    skip "short-5s.wav"
  fi

  # ─── numbers-dates.wav ────────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/numbers-dates.wav"; then
    info "Generating numbers-dates.wav (macOS say)"
    say -o "$SPEECH_DIR/numbers-dates.aiff" \
      "The meeting is at 3:45 PM on January 15th, 2025. Please call 555-0123 to confirm your attendance. The office is located at 742 Evergreen Terrace."
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/numbers-dates.aiff" "$SPEECH_DIR/numbers-dates.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/numbers-dates.aiff"
    else
      mv "$SPEECH_DIR/numbers-dates.aiff" "$SPEECH_DIR/numbers-dates.wav"
    fi
    ok "numbers-dates.wav"
  else
    skip "numbers-dates.wav"
  fi

  # ─── technical-terms.wav ──────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/technical-terms.wav"; then
    info "Generating technical-terms.wav (macOS say)"
    say -o "$SPEECH_DIR/technical-terms.aiff" \
      "Kubernetes orchestrates containerized microservices across distributed clusters. Each pod contains one or more containers sharing network namespaces and storage volumes. The control plane manages scheduling, scaling, and self-healing through reconciliation loops."
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/technical-terms.aiff" "$SPEECH_DIR/technical-terms.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/technical-terms.aiff"
    else
      mv "$SPEECH_DIR/technical-terms.aiff" "$SPEECH_DIR/technical-terms.wav"
    fi
    ok "technical-terms.wav"
  else
    skip "technical-terms.wav"
  fi

  # ─── clean-narration.wav ──────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/clean-narration.wav"; then
    info "Generating clean-narration.wav (macOS say)"
    say -o "$SPEECH_DIR/clean-narration.aiff" \
      "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness."
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/clean-narration.aiff" "$SPEECH_DIR/clean-narration.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/clean-narration.aiff"
    else
      mv "$SPEECH_DIR/clean-narration.aiff" "$SPEECH_DIR/clean-narration.wav"
    fi
    ok "clean-narration.wav"
  else
    skip "clean-narration.wav"
  fi

  # ─── long-narration.wav ───────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/long-narration.wav"; then
    info "Generating long-narration.wav (macOS say)"
    say -o "$SPEECH_DIR/long-narration.aiff" \
      "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation."
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/long-narration.aiff" "$SPEECH_DIR/long-narration.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/long-narration.aiff"
    else
      mv "$SPEECH_DIR/long-narration.aiff" "$SPEECH_DIR/long-narration.wav"
    fi
    ok "long-narration.wav"
  else
    skip "long-narration.wav"
  fi

  # ─── spanish-speech.wav ───────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/spanish-speech.wav"; then
    # Try Spanish voice, fall back to default
    local SPANISH_VOICE="Paulina"
    if say -v "$SPANISH_VOICE" "" 2>/dev/null; then
      info "Generating spanish-speech.wav (macOS say -v $SPANISH_VOICE)"
      say -v "$SPANISH_VOICE" -o "$SPEECH_DIR/spanish-speech.aiff" \
        "Buenos días, bienvenidos a la presentación de hoy. Vamos a hablar sobre la inteligencia artificial y su impacto en la sociedad moderna."
    else
      info "Generating spanish-speech.wav (macOS say, default voice)"
      say -o "$SPEECH_DIR/spanish-speech.aiff" \
        "Buenos días, bienvenidos a la presentación de hoy. Vamos a hablar sobre la inteligencia artificial y su impacto en la sociedad moderna."
    fi
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/spanish-speech.aiff" "$SPEECH_DIR/spanish-speech.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/spanish-speech.aiff"
    else
      mv "$SPEECH_DIR/spanish-speech.aiff" "$SPEECH_DIR/spanish-speech.wav"
    fi
    ok "spanish-speech.wav"
  else
    skip "spanish-speech.wav"
  fi

  # ─── accented-indian.wav ──────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/accented-indian.wav"; then
    local INDIAN_VOICE="Rishi"
    if say -v "$INDIAN_VOICE" "" 2>/dev/null; then
      info "Generating accented-indian.wav (macOS say -v $INDIAN_VOICE)"
      say -v "$INDIAN_VOICE" -o "$SPEECH_DIR/accented-indian.aiff" \
        "The software development lifecycle includes planning, analysis, design, implementation, testing, and maintenance phases. Each phase has specific deliverables and review processes."
    else
      info "Generating accented-indian.wav (macOS say, default voice)"
      say -o "$SPEECH_DIR/accented-indian.aiff" \
        "The software development lifecycle includes planning, analysis, design, implementation, testing, and maintenance phases. Each phase has specific deliverables and review processes."
    fi
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/accented-indian.aiff" "$SPEECH_DIR/accented-indian.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/accented-indian.aiff"
    else
      mv "$SPEECH_DIR/accented-indian.aiff" "$SPEECH_DIR/accented-indian.wav"
    fi
    ok "accented-indian.wav"
  else
    skip "accented-indian.wav"
  fi

  # ─── accented-british.wav ─────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/accented-british.wav"; then
    local BRITISH_VOICE="Daniel"
    if say -v "$BRITISH_VOICE" "" 2>/dev/null; then
      info "Generating accented-british.wav (macOS say -v $BRITISH_VOICE)"
      say -v "$BRITISH_VOICE" -o "$SPEECH_DIR/accented-british.aiff" \
        "The quarterly financial results exceeded our expectations. Revenue grew by twelve percent year over year, driven primarily by strong performance in the cloud services division."
    else
      info "Generating accented-british.wav (macOS say, default voice)"
      say -o "$SPEECH_DIR/accented-british.aiff" \
        "The quarterly financial results exceeded our expectations. Revenue grew by twelve percent year over year, driven primarily by strong performance in the cloud services division."
    fi
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/accented-british.aiff" "$SPEECH_DIR/accented-british.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/accented-british.aiff"
    else
      mv "$SPEECH_DIR/accented-british.aiff" "$SPEECH_DIR/accented-british.wav"
    fi
    ok "accented-british.wav"
  else
    skip "accented-british.wav"
  fi

  # ─── phone-call.wav ──────────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/phone-call.wav"; then
    info "Generating phone-call.wav (8kHz downsampled)"
    say -o "$SPEECH_DIR/phone-call.aiff" \
      "Thank you for calling customer support. Your account balance is four hundred and twenty three dollars and seventeen cents. Your next payment of fifty dollars is due on April first."
    if has_cmd afconvert; then
      # Convert to 8kHz WAV to simulate phone-quality audio
      afconvert "$SPEECH_DIR/phone-call.aiff" "$SPEECH_DIR/phone-call.wav" -d LEI16 -f WAVE -r 8000
      rm -f "$SPEECH_DIR/phone-call.aiff"
    else
      mv "$SPEECH_DIR/phone-call.aiff" "$SPEECH_DIR/phone-call.wav"
    fi
    ok "phone-call.wav"
  else
    skip "phone-call.wav"
  fi

  # ─── fast-speech.wav ─────────────────────────────────────────────────────
  if should_generate "$SPEECH_DIR/fast-speech.wav"; then
    info "Generating fast-speech.wav (high speech rate)"
    say -r 260 -o "$SPEECH_DIR/fast-speech.aiff" \
      "In machine learning, gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient."
    if has_cmd afconvert; then
      afconvert "$SPEECH_DIR/fast-speech.aiff" "$SPEECH_DIR/fast-speech.wav" -d LEI16 -f WAVE
      rm -f "$SPEECH_DIR/fast-speech.aiff"
    else
      mv "$SPEECH_DIR/fast-speech.aiff" "$SPEECH_DIR/fast-speech.wav"
    fi
    ok "fast-speech.wav"
  else
    skip "fast-speech.wav"
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════
verify_corpus() {
  echo ""
  echo "═══ Corpus Verification ═══"
  echo ""

  local vision_expected=(
    "receipt-grocery.jpg"
    "receipt-restaurant.jpg"
    "invoice-standard.pdf"
    "business-card.jpg"
    "screenshot-code.png"
    "table-financial.png"
    "multilang-french.jpg"
    "multilang-japanese.jpg"
    "book-page.jpg"
    "menu-restaurant.jpg"
    "prescription-label.jpg"
    "rotated-scan.jpg"
    "low-quality-scan.jpg"
    "mixed-layout-newsletter.pdf"
    "multipage-report.pdf"
  )

  local speech_expected=(
    "short-5s.wav"
    "numbers-dates.wav"
    "technical-terms.wav"
    "clean-narration.wav"
    "long-narration.wav"
    "spanish-speech.wav"
    "accented-indian.wav"
    "accented-british.wav"
    "phone-call.wav"
    "fast-speech.wav"
  )

  local vision_present=0
  local vision_missing=0
  local speech_present=0
  local speech_missing=0

  echo "  Vision OCR files:"
  printf "  %-35s %-10s %-10s\n" "File" "Image" "GroundTruth"
  printf "  %-35s %-10s %-10s\n" "---" "---" "---"
  for f in "${vision_expected[@]}"; do
    local base="${f%.*}"
    local img_status="MISSING"
    local txt_status="MISSING"
    [[ -f "$VISION_DIR/$f" ]] && img_status="OK" && vision_present=$((vision_present + 1))
    [[ -f "$VISION_DIR/$base.txt" ]] && txt_status="OK"
    [[ "$img_status" == "MISSING" ]] && vision_missing=$((vision_missing + 1))
    printf "  %-35s %-10s %-10s\n" "$f" "$img_status" "$txt_status"
  done

  echo ""
  echo "  Speech files:"
  printf "  %-35s %-10s %-10s\n" "File" "Audio" "GroundTruth"
  printf "  %-35s %-10s %-10s\n" "---" "---" "---"
  for f in "${speech_expected[@]}"; do
    local base="${f%.*}"
    local audio_status="MISSING"
    local txt_status="MISSING"
    [[ -f "$SPEECH_DIR/$f" ]] && audio_status="OK" && speech_present=$((speech_present + 1))
    [[ -f "$SPEECH_DIR/$base.txt" ]] && txt_status="OK"
    [[ "$audio_status" == "MISSING" ]] && speech_missing=$((speech_missing + 1))
    printf "  %-35s %-10s %-10s\n" "$f" "$audio_status" "$txt_status"
  done

  echo ""
  echo "  Summary: Vision ${vision_present}/${#vision_expected[@]} present, Speech ${speech_present}/${#speech_expected[@]} present"

  if [[ $vision_missing -eq 0 && $speech_missing -eq 0 ]]; then
    echo "  All corpus files present."
    return 0
  else
    echo "  Run ./Scripts/generate-test-corpus.sh to generate missing files."
    return 1
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Vision/Speech Test Corpus Generator                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [[ "$VERIFY_ONLY" == "true" ]]; then
  verify_corpus
  exit $?
fi

if [[ "$SPEECH_ONLY" != "true" ]]; then
  generate_vision_corpus
fi

if [[ "$VISION_ONLY" != "true" ]]; then
  generate_speech_corpus
fi

verify_corpus
echo ""
echo "Done."
