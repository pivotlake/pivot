#!/usr/bin/env bash
set -euo pipefail

export PATH="$PATH:$HOME/.cargo/bin"

source_path="${1:-${TPCH_FLAT_SOURCE:-$HOME/tpch-flat}}"
scale_factor="${TPCH_SCALE_FACTOR:-1}"
batch_rows="${TPCH_FLAT_BATCH_ROWS:-65536}"
row_group_rows="${TPCH_FLAT_ROW_GROUP_ROWS:-1048576}"

if [[ -s "${source_path%/}/tpch_flat.parquet" ]]; then
    printf '%s\n' "$source_path"
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmarks_dir="$(cd "$script_dir/.." && pwd)"
flat_dir="${source_path%/}/flat"
parquet_file="${flat_dir%/}/tpch_flat.parquet"
marker_file="${source_path%/}/.tpch-flat.sf"
tpchgen_dir="${TPCHGEN_SOURCE:-${source_path%/}/tpchgen-sf-${scale_factor}}"
tpchgen_marker="${tpchgen_dir%/}/.tpchgen.sf"
marker_value="sf=${scale_factor};row_group_rows=${row_group_rows}"

case "$parquet_file:$tpchgen_dir:$benchmarks_dir" in
    *"'"*) echo "error: paths must not contain a single quote: $source_path" >&2; exit 1 ;;
esac

if [[ -s "$parquet_file" && -f "$marker_file" && "$(<"$marker_file")" == "$marker_value" ]]; then
    printf '%s\n' "$flat_dir"
    exit 0
fi

ensure_tpchgen() {
    if command -v tpchgen-cli >/dev/null 2>&1; then
        return 0
    fi
    if [[ "${TPCHGEN_INSTALL:-1}" != "1" ]]; then
        echo "error: tpchgen-cli not found on PATH" >&2
        echo "install it with: cargo install tpchgen-cli" >&2
        exit 1
    fi
    command -v cargo >/dev/null 2>&1 || {
        echo "error: tpchgen-cli not found, and cargo is unavailable for installation" >&2
        exit 1
    }
    echo "installing tpchgen-cli via cargo" >&2
    cargo install tpchgen-cli
}

generate_tpchgen_parquet() {
    local missing=0
    local table
    for table in customer lineitem nation orders part partsupp region supplier; do
        [[ -s "${tpchgen_dir%/}/${table}.parquet" ]] || missing=1
    done
    if [[ $missing -eq 0 && -f "$tpchgen_marker" && "$(<"$tpchgen_marker")" == "$scale_factor" ]]; then
        return 0
    fi

    ensure_tpchgen
    mkdir -p "$tpchgen_dir"
    echo "generating TPCH normalized parquet with tpchgen-cli at scale factor $scale_factor -> $tpchgen_dir" >&2

    local args=(
        --scale-factor "$scale_factor"
        --format=parquet
        --output-dir "$tpchgen_dir"
    )
    if [[ -n "${TPCHGEN_THREADS:-}" ]]; then
        args+=(--num-threads "$TPCHGEN_THREADS")
    fi
    if [[ -n "${TPCHGEN_PARQUET_ROW_GROUP_BYTES:-}" ]]; then
        args+=(--parquet-row-group-bytes "$TPCHGEN_PARQUET_ROW_GROUP_BYTES")
    fi

    tpchgen-cli "${args[@]}" >&2
    printf '%s\n' "$scale_factor" > "$tpchgen_marker"
}

mkdir -p "$source_path"
mkdir -p "$flat_dir"
generate_tpchgen_parquet

tmp_parquet="${parquet_file}.tmp"
rm -f "$tmp_parquet"

echo "materializing wide TPCH flat parquet -> $parquet_file" >&2
cargo run \
    --manifest-path "$benchmarks_dir/Cargo.toml" \
    --release \
    --bin tpch-flatten \
    -- "$tpchgen_dir" "$tmp_parquet" \
    --batch-rows "$batch_rows" \
    --row-group-rows "$row_group_rows" >&2
mv "$tmp_parquet" "$parquet_file"
printf '%s\n' "$marker_value" > "$marker_file"
printf '%s\n' "$flat_dir"
