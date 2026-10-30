#!/usr/bin/env bash
# Compile the manuscript and (re)build the Overleaf-ready zip: paper/manuscript_latex.zip
set -euo pipefail
cd "$(dirname "$0")/../paper"
latexmk -pdf -interaction=nonstopmode main.tex
rm -f manuscript_latex.zip
zip -r manuscript_latex.zip main.tex references.tex sections tables figures -x '*.aux' '*.log'
echo "Built paper/main.pdf and paper/manuscript_latex.zip"
