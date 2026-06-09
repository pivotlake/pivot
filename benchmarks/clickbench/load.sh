#!/usr/bin/env bash
set -euo pipefail

source_path="${1:-${HITS_SOURCE:-$HOME/hits}}"
url="${CLICKBENCH_PARQUET_URL:-https://datasets.clickhouse.com/hits_compatible/hits.parquet}"

download_parquet() {
    local target="$1"
    local tmp="${target}.tmp"

    mkdir -p "$(dirname "$target")"
    rm -f "$tmp"

    echo "downloading ClickBench parquet -> $target" >&2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        echo "error: neither curl nor wget is available to download $url" >&2
        exit 1
    fi

    mv "$tmp" "$target"
}

has_nonempty_parquet() {
    local dir="$1"
    local files=()
    shopt -s nullglob
    files=("${dir%/}"/*.parquet)
    shopt -u nullglob
    for file in "${files[@]}"; do
        [[ -s "$file" ]] && return 0
    done
    return 1
}

case "$source_path" in
    *'*'*|*'?'*|*'['*)
        if compgen -G "$source_path" >/dev/null; then
            printf '%s\n' "$source_path"
            exit 0
        fi
        echo "error: ClickBench source glob matched no files: $source_path" >&2
        echo "pass a directory or .parquet file path if load.sh should download the dataset" >&2
        exit 1
        ;;
esac

if [[ -f "$source_path" ]]; then
    if [[ "$source_path" != *.parquet ]]; then
        echo "error: ClickBench source file is not a parquet file: $source_path" >&2
        exit 1
    fi
    if [[ ! -s "$source_path" ]]; then
        download_parquet "$source_path"
    fi
    printf '%s\n' "$source_path"
    exit 0
fi

if [[ "$source_path" == *.parquet ]]; then
    [[ -s "$source_path" ]] || download_parquet "$source_path"
    printf '%s\n' "$source_path"
    exit 0
fi

mkdir -p "$source_path"
if ! has_nonempty_parquet "$source_path"; then
    download_parquet "${source_path%/}/hits.parquet"
fi

printf '%s\n' "$source_path"
