#!/bin/bash

# This benchmarking script is responsible for:
# 1) Making the data directory if needed
# 2) Download the FastText embeddings if not available locally
# 3) Run the FAISS HNSW benchmarking
# 4) Run the Rust Annoy Index logic

# Make the data folder if not exists
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"
DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

# Download the wikidata data file if not exists
WIKIDATA_FILE="${DATA_DIR}/wikidata.vec"
WIKIDATA_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
if [ ! -f "$WIKIDATA_FILE" ]; then
    echo "$WIKIDATA_FILE does not exist. Downloading..."
    curl -o temporary.zip "$WIKIDATA_URL"
    unzip temporary.zip
    mv wiki-news-300d-1M.vec "$WIKIDATA_FILE"
    rm -rf temporary.zip
else
    echo "$WIKIDATA_FILE already exists."
fi

# Run the faiss benchmarking
FAISS_FILE="${SCRIPT_DIR}/faiss_run.py"
python3 "$FAISS_FILE" --input-vec "$WIKIDATA_FILE" --data-dir "$DATA_DIR"

# Benchmark our Rust index
CWD_TO_PRESERVE=$(pwd)
cd "$ROOT_DIR"
cargo build --release
./target/release/ann "$DATA_DIR" "$WIKIDATA_FILE"
cd "$CWD_TO_PRESERVE"
