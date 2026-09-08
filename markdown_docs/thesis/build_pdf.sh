#!/usr/bin/env bash
# Build a PDF from a Markdown source via pandoc + XeLaTeX.
#
#   ./build_pdf.sh Thesis_outline/Thesis_outline_EN.md
#   ./build_pdf.sh Thesis_outline/Scaletta_tesi.md /tmp/scaletta.pdf
#
# Output defaults to <input>.pdf next to the source.
set -euo pipefail

src="${1:?usage: build_pdf.sh <input.md> [output.pdf]}"
out="${2:-${src%.md}.pdf}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pandoc "$src" \
    --from=markdown \
    --pdf-engine=xelatex \
    --lua-filter="$here/unicode_fallback.lua" \
    --include-in-header="$here/preamble.tex" \
    --toc --toc-depth=3 \
    --number-sections \
    -V papersize=a4 \
    -V geometry:margin=2.5cm \
    -V fontsize=11pt \
    -V mainfont="DejaVu Serif" \
    -V sansfont="DejaVu Sans" \
    -V monofont="DejaVu Sans Mono" \
    -V colorlinks=true \
    -V linkcolor=RoyalBlue \
    -V urlcolor=RoyalBlue \
    -o "$out"

echo "wrote $out"
