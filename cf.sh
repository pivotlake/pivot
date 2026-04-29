#!/bin/bash
# cache-funnel.sh — Parse perf stat output and display L1→L2→L3→DRAM funnel
# Usage: cat perf_output.txt | ./cache-funnel.sh
#   or:  ./cache-funnel.sh < perf_output.txt

PERF='L1-dcache-loads,L1-dcache-load-misses,l2_cache_req_stat.dc_access_in_l2,l2_cache_req_stat.dc_hit_in_l2,ls_dmnd_fills_from_sys.local_ccx,ls_dmnd_fills_from_sys.near_cache,ls_dmnd_fills_from_sys.far_cache,ls_dmnd_fills_from_sys.dram_io_near,ls_dmnd_fills_from_sys.dram_io_far,l2_pf_miss_l2_hit_l3.all,l2_pf_miss_l2_l3.all,ls_hw_pf_dc_fills.local_l2,ls_hw_pf_dc_fills.local_ccx,ls_hw_pf_dc_fills.near_cache,ls_hw_pf_dc_fills.far_cache,ls_hw_pf_dc_fills.dram_io_near,ls_hw_pf_dc_fills.dram_io_far,ls_sw_pf_dc_fills.local_l2,ls_sw_pf_dc_fills.local_ccx,ls_sw_pf_dc_fills.near_cache,ls_sw_pf_dc_fills.far_cache,ls_sw_pf_dc_fills.dram_io_near,ls_sw_pf_dc_fills.dram_io_far,ic_cache_fill_l2,ic_cache_fill_sys,de_dis_dispatch_token_stalls1.int_phy_reg_file_rsrc_stall,de_dis_dispatch_token_stalls1.load_queue_rsrc_stall,de_dis_dispatch_token_stalls1.store_queue_rsrc_stall,de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall,de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall,de_dis_dispatch_token_stalls1.fp_flush_recovery_stall,de_dis_dispatch_token_stalls1.taken_brnch_buffer_rsrc,de_dis_dispatch_token_stalls2.retire_token_stall,de_dis_dispatch_token_stalls2.int_sch0_token_stall,de_dis_dispatch_token_stalls2.int_sch1_token_stall,de_dis_dispatch_token_stalls2.int_sch2_token_stall,de_dis_dispatch_token_stalls2.int_sch3_token_stall,cpu/de_no_dispatch_per_slot.backend_stalls,cmask=0x6/,cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/,cpu/de_no_dispatch_per_slot.smt_contention,cmask=0x6/,de_op_queue_empty,cpu/event=0xAE,umask=0x08/,cycles,amd_umc/umc_cas_cmd.rd/,amd_umc/umc_cas_cmd.wr/'

err_on_missing=0
for arg in "$@"; do
    case "$arg" in
        --err-on-missing) err_on_missing=1 ;;
    esac
done

input=$(cat)

parse_val() {
    echo "$1" | sed 's/^ *//;s/ .*//' | tr -d ','
}

l1_loads=0; l1_misses=0; l2_access=0; l2_hits=0
dmnd_local_ccx=0; dmnd_near_cache=0; dmnd_far_cache=0; dmnd_dram=0; dmnd_dram_far=0
l2pf_hit_l3=0; l2pf_miss_l3=0
hwpf_from_l2=0; hwpf_from_l3=0; hwpf_near_cache=0; hwpf_far_cache=0; hwpf_from_dram=0; hwpf_dram_far=0
swpf_from_l2=0; swpf_from_l3=0; swpf_near_cache=0; swpf_far_cache=0; swpf_from_dram=0; swpf_dram_far=0
ic_fill_l2=0; ic_fill_sys=0
# stalls - backend
stall_backend=0
stall_int_reg=0; stall_load_q=0; stall_store_q=0
stall_fp_reg=0; stall_fp_sch=0; stall_fp_flush=0
stall_taken_br=0; stall_retire=0
stall_int_sch0=0; stall_int_sch1=0; stall_int_sch2=0; stall_int_sch3=0
stall_bitcoin=0
# stalls - frontend
stall_frontend=0; stall_opq_empty=0; stall_smt=0
umc_cas_rd=0; umc_cas_wr=0
elapsed_s=0; perf_window_ms=0

perf_window_ms=$(echo "$input" | grep -oP 'perf window: \K[0-9.]+' | awk '{s+=$1} END {printf "%.1f", s}')

while IFS= read -r line; do
    case "$line" in
        *L1-dcache-loads*)                         l1_loads=$(parse_val "$line") ;;
        *L1-dcache-load-misses*)                   l1_misses=$(parse_val "$line") ;;
        *dc_access_in_l2*)                         l2_access=$(parse_val "$line") ;;
        *dc_hit_in_l2*)                            l2_hits=$(parse_val "$line") ;;
        *ls_dmnd_fills_from_sys.local_ccx*)        dmnd_local_ccx=$(parse_val "$line") ;;
        *ls_dmnd_fills_from_sys.near_cache*)       dmnd_near_cache=$(parse_val "$line") ;;
        *ls_dmnd_fills_from_sys.far_cache*)        dmnd_far_cache=$(parse_val "$line") ;;
        *ls_dmnd_fills_from_sys.dram_io_near*)     dmnd_dram=$(parse_val "$line") ;;
        *ls_dmnd_fills_from_sys.dram_io_far*)      dmnd_dram_far=$(parse_val "$line") ;;
        *l2_pf_miss_l2_hit_l3.all*)                l2pf_hit_l3=$(parse_val "$line") ;;
        *l2_pf_miss_l2_l3.all*)                    l2pf_miss_l3=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.local_l2*)              hwpf_from_l2=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.local_ccx*)             hwpf_from_l3=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.near_cache*)            hwpf_near_cache=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.far_cache*)             hwpf_far_cache=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.dram_io_near*)          hwpf_from_dram=$(parse_val "$line") ;;
        *ls_hw_pf_dc_fills.dram_io_far*)           hwpf_dram_far=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.local_l2*)              swpf_from_l2=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.local_ccx*)             swpf_from_l3=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.near_cache*)            swpf_near_cache=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.far_cache*)             swpf_far_cache=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.dram_io_near*)          swpf_from_dram=$(parse_val "$line") ;;
        *ls_sw_pf_dc_fills.dram_io_far*)           swpf_dram_far=$(parse_val "$line") ;;
        *ic_cache_fill_l2*)                         ic_fill_l2=$(parse_val "$line") ;;
        *ic_cache_fill_sys*)                        ic_fill_sys=$(parse_val "$line") ;;
        *de_no_dispatch_per_slot.backend_stalls*)  stall_backend=$(parse_val "$line") ;;
        *de_no_dispatch_per_slot.no_ops_from_frontend*) stall_frontend=$(parse_val "$line") ;;
        *de_no_dispatch_per_slot.smt_contention*)  stall_smt=$(parse_val "$line") ;;
        *de_op_queue_empty*)                       stall_opq_empty=$(parse_val "$line") ;;
        *int_phy_reg_file_rsrc_stall*)             stall_int_reg=$(parse_val "$line") ;;
        *load_queue_rsrc_stall*)                   stall_load_q=$(parse_val "$line") ;;
        *store_queue_rsrc_stall*)                  stall_store_q=$(parse_val "$line") ;;
        *fp_reg_file_rsrc_stall*)                  stall_fp_reg=$(parse_val "$line") ;;
        *fp_sch_rsrc_stall*)                       stall_fp_sch=$(parse_val "$line") ;;
        *fp_flush_recovery_stall*)                 stall_fp_flush=$(parse_val "$line") ;;
        *taken_brnch_buffer_rsrc*)                 stall_taken_br=$(parse_val "$line") ;;
        *retire_token_stall*)                      stall_retire=$(parse_val "$line") ;;
        *int_sch0_token_stall*)                    stall_int_sch0=$(parse_val "$line") ;;
        *int_sch1_token_stall*)                    stall_int_sch1=$(parse_val "$line") ;;
        *int_sch2_token_stall*)                    stall_int_sch2=$(parse_val "$line") ;;
        *int_sch3_token_stall*)                    stall_int_sch3=$(parse_val "$line") ;;
        *event=0xAE*umask=0x08*|*event=0xae*umask=0x08*) stall_bitcoin=$(parse_val "$line") ;;
        *umc_cas_cmd.rd*)                          umc_cas_rd=$(parse_val "$line") ;;
        *umc_cas_cmd.wr*)                          umc_cas_wr=$(parse_val "$line") ;;
        *seconds\ time\ elapsed*)                  elapsed_s=$(echo "$line" | sed 's/^ *//;s/ .*//' | tr -d ',') ;;
        *cycles*)                                  cycles_line=$(parse_val "$line") ;;
    esac
done < <(echo "$input" | sed -n '/Performance counter stats/,$ p')

if [ "$err_on_missing" -eq 1 ]; then
    missing=()
    IFS=',' read -ra expected <<< "$PERF"
    for evt in "${expected[@]}"; do
        # strip PMU wrappers (amd_umc/, cpu/) and trailing / for matching
        match="${evt#amd_umc/}"
        match="${match#cpu/}"
        match="${match%/}"
        # for raw events or cmask, match the core event name
        match=$(echo "$match" | sed 's/,cmask=[^,]*//' | sed 's/,umask=[^,]*//' | sed 's/event=[^,]*//')
        match=$(echo "$match" | sed 's/^,//;s/,$//')
        if [ -z "$match" ]; then continue; fi  # skip raw-only events
        if ! echo "$input" | grep -q "$match"; then
            missing+=("$evt")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: missing events in perf output:" >&2
        for m in "${missing[@]}"; do
            echo "  $m" >&2
        done
        exit 1
    fi
fi

l1_hits=$((l1_loads - l1_misses))
l2_misses=$((l2_access - l2_hits))
dmnd_dram_total=$((dmnd_dram + dmnd_dram_far))
l3_dmnd_hits=$((dmnd_local_ccx + dmnd_near_cache + dmnd_far_cache))
l3_dmnd_total=$((l3_dmnd_hits + dmnd_dram_total))
l1_pf_to_l2=$((l2_access - l1_misses))
l2pf_total=$((l2pf_hit_l3 + l2pf_miss_l3))
hwpf_l3_total=$((hwpf_from_l3 + hwpf_near_cache + hwpf_far_cache))
hwpf_dram_total=$((hwpf_from_dram + hwpf_dram_far))
hwpf_total=$((hwpf_from_l2 + hwpf_l3_total + hwpf_dram_total))
swpf_l3_total=$((swpf_from_l3 + swpf_near_cache + swpf_far_cache))
swpf_dram_total=$((swpf_from_dram + swpf_dram_far))
swpf_total=$((swpf_from_l2 + swpf_l3_total + swpf_dram_total))

pct() {
    if [ "$2" -eq 0 ]; then echo "0.0"; return; fi
    awk "BEGIN { printf \"%.1f\", ($1 / $2) * 100 }"
}

fmt() {
    if [ "$1" -ge 1000000000 ]; then
        awk "BEGIN { printf \"%.2fB\", $1 / 1000000000 }"
    elif [ "$1" -ge 1000000 ]; then
        awk "BEGIN { printf \"%.1fM\", $1 / 1000000 }"
    elif [ "$1" -ge 1000 ]; then
        awk "BEGIN { printf \"%.1fK\", $1 / 1000 }"
    else
        echo "$1"
    fi
}

to_gb() {
    awk "BEGIN { printf \"%.1f\", $1 * 64 / 1073741824 }"
}

G='\033[32m'; R='\033[31m'; Y='\033[33m'; C='\033[36m'; M='\033[35m'; B='\033[1m'; D='\033[0m'; W='\033[37m'

hit_pct=$(pct $l1_hits $l1_loads)
miss_pct=$(pct $l1_misses $l1_loads)
echo ""
echo -e "  ${B}╔══════════════════════════════════════════════════╗${D}"
echo -e "  ${B}║              CACHE HIERARCHY FUNNEL              ║${D}"
echo -e "  ${B}╚══════════════════════════════════════════════════╝${D}"
echo ""
echo -e "  ${B}L1 Data Cache${D}          $(fmt $l1_loads) accesses"
echo -e "  ┌──────────────────────────────────────────────────┐"
echo -e "  │  ${G}HIT  $(fmt $l1_hits) (${hit_pct}%)${D}"
echo -e "  │  ${R}MISS $(fmt $l1_misses) (${miss_pct}%)${D}"
if [ "$hwpf_total" -gt 0 ]; then
echo -e "  │  ${M}HW PF $(fmt $hwpf_total) HW prefetch fills into L1${D}"
echo -e "  │  ${M}  ├─ $(fmt $hwpf_from_l2) from L2${D}"
echo -e "  │  ${M}  ├─ $(fmt $hwpf_l3_total) from L3/CCX${D}"
echo -e "  │  ${M}  └─ $(fmt $hwpf_dram_total) from DRAM${D}"
fi
if [ "$swpf_total" -gt 0 ]; then
echo -e "  │  ${C}SW PF $(fmt $swpf_total) SW prefetch fills into L1${D}"
echo -e "  │  ${C}  ├─ $(fmt $swpf_from_l2) from L2${D}"
echo -e "  │  ${C}  ├─ $(fmt $swpf_l3_total) from L3/CCX${D}"
echo -e "  │  ${C}  └─ $(fmt $swpf_dram_total) from DRAM${D}"
fi
echo -e "  └───────────────────────┬──────────────────────────┘"
echo -e "                          │ $(fmt $l1_misses) demand misses"
if [ "$l1_pf_to_l2" -gt 0 ] 2>/dev/null; then
echo -e "                          │ + $(fmt $l1_pf_to_l2) L1 prefetch misses"
fi
echo -e "                          ▼"

hit_pct=$(pct $l2_hits $l2_access)
miss_pct=$(pct $l2_misses $l2_access)
echo -e "  ${B}L2 Cache${D}               $(fmt $l2_access) accesses (excl. L2 prefetch)"
echo -e "  ┌──────────────────────────────────────────────────┐"
echo -e "  │  ${G}HIT  $(fmt $l2_hits) (${hit_pct}%)${D}"
echo -e "  │  ${R}MISS $(fmt $l2_misses) (${miss_pct}%)${D}"
echo -e "  └───────────────────────┬──────────────────────────┘"
echo -e "                          │ $(fmt $l2_misses) demand misses"
echo -e "                          ▼"

hit_pct=$(pct $l3_dmnd_hits $l3_dmnd_total)
miss_pct=$(pct $dmnd_dram_total $l3_dmnd_total)
echo -e "  ${B}L3 Cache${D}               $(fmt $l3_dmnd_total) demand accesses"
echo -e "  ┌──────────────────────────────────────────────────┐"
echo -e "  │  ${G}HIT  $(fmt $l3_dmnd_hits) (${hit_pct}%)${D}"
if [ "$dmnd_near_cache" -gt 0 ] || [ "$dmnd_far_cache" -gt 0 ]; then
echo -e "  │  ${G}  ├─ $(fmt $dmnd_local_ccx) local CCX${D}"
echo -e "  │  ${G}  ├─ $(fmt $dmnd_near_cache) near CCX${D}"
echo -e "  │  ${G}  └─ $(fmt $dmnd_far_cache) far CCX${D}"
fi
echo -e "  │  ${R}MISS $(fmt $dmnd_dram_total) (${miss_pct}%)${D}"
if [ "$dmnd_dram_far" -gt 0 ]; then
echo -e "  │  ${R}  ├─ $(fmt $dmnd_dram) near DRAM${D}"
echo -e "  │  ${R}  └─ $(fmt $dmnd_dram_far) far DRAM${D}"
fi
if [ "$l2pf_total" -gt 0 ]; then
echo -e "  │  ${M}PF   $(fmt $l2pf_total) L2 prefetch requests${D}"
echo -e "  │  ${M}  ├─ $(fmt $l2pf_hit_l3) hit L3${D}"
echo -e "  │  ${M}  └─ $(fmt $l2pf_miss_l3) miss L3 → DRAM${D}"
fi
echo -e "  └───────────────────────┬──────────────────────────┘"
echo -e "                          │ $(fmt $dmnd_dram_total) demand misses"
if [ "$l2pf_miss_l3" -gt 0 ]; then
echo -e "                          │ + $(fmt $l2pf_miss_l3) L2 prefetch misses"
fi
if [ "$hwpf_dram_total" -gt 0 ]; then
echo -e "                          │ + $(fmt $hwpf_dram_total) HW prefetch misses"
fi
if [ "$swpf_dram_total" -gt 0 ]; then
echo -e "                          │ + $(fmt $swpf_dram_total) SW prefetch misses"
fi
if [ "$ic_fill_sys" -gt 0 ]; then
echo -e "                          │ + $(fmt $ic_fill_sys) IC fills (L3/DRAM)"
fi
echo -e "                          ▼"

core_dram_reads=$((dmnd_dram_total + l2pf_miss_l3 + hwpf_dram_total + swpf_dram_total + ic_fill_sys))
echo -e "  ${B}DRAM${D}                   $(fmt $core_dram_reads) core read fills"
echo -e "  ┌──────────────────────────────────────────────────┐"
core_rd_gb=$(to_gb $core_dram_reads)
dmnd_gb=$(to_gb $dmnd_dram_total)
l2pf_gb=$(to_gb $l2pf_miss_l3)
hwpf_gb=$(to_gb $hwpf_dram_total)
swpf_gb=$(to_gb $swpf_dram_total)
ic_gb=$(to_gb $ic_fill_sys)
echo -e "  │  ${Y}~${core_rd_gb} GB read (× 64B cachelines)${D}"
echo -e "  │  ${Y}  demand:  ${dmnd_gb} GB  ($(fmt $dmnd_dram_total))${D}"
echo -e "  │  ${Y}  L2 pf:   ${l2pf_gb} GB  ($(fmt $l2pf_miss_l3))${D}"
echo -e "  │  ${Y}  HW pf:   ${hwpf_gb} GB  ($(fmt $hwpf_dram_total))${D}"
echo -e "  │  ${Y}  SW pf:   ${swpf_gb} GB  ($(fmt $swpf_dram_total))${D}"
echo -e "  │  ${Y}  IC:      ${ic_gb} GB  ($(fmt $ic_fill_sys))${D}"

# UMC stats
if [ "$umc_cas_rd" -gt 0 ] || [ "$umc_cas_wr" -gt 0 ]; then
    umc_rd_gb=$(to_gb $umc_cas_rd)
    umc_wr_gb=$(to_gb $umc_cas_wr)
    umc_total=$((umc_cas_rd + umc_cas_wr))
    umc_total_gb=$(to_gb $umc_total)
    echo -e "  │  ${W}──────────────────────────────────────────────${D}"
    echo -e "  │  ${B}UMC${D}  $(fmt $umc_cas_rd) reads (${umc_rd_gb} GB)"
    echo -e "  │       $(fmt $umc_cas_wr) writes (${umc_wr_gb} GB)"
    echo -e "  │       $(fmt $umc_total) total (${umc_total_gb} GB)"
    coverage_pct=$(pct $core_dram_reads $umc_cas_rd)
    echo -e "  │  ${C}core reads / UMC reads: ${coverage_pct}%${D}"
fi

# Bandwidth
bw_time=""
bw_label=""
if awk "BEGIN { exit ($perf_window_ms > 0) ? 0 : 1 }" 2>/dev/null; then
    bw_time=$perf_window_ms
    bw_label="perf window"
    secs=$(awk "BEGIN { printf \"%.4f\", $perf_window_ms / 1000 }")
elif awk "BEGIN { exit ($elapsed_s > 0) ? 0 : 1 }" 2>/dev/null; then
    bw_time=$elapsed_s
    bw_label="elapsed"
    secs=$elapsed_s
else
    secs=""
fi

if [ -n "$secs" ]; then
    echo -e "  │  ${W}──────────────────────────────────────────────${D}"
    core_rd_gbps=$(awk "BEGIN { printf \"%.1f\", ($core_dram_reads * 64 / 1073741824) / $secs }")
    dmnd_gbps=$(awk "BEGIN { printf \"%.1f\", ($dmnd_dram_total * 64 / 1073741824) / $secs }")
    if [ "$umc_cas_rd" -gt 0 ] || [ "$umc_cas_wr" -gt 0 ]; then
        umc_rd_gbps=$(awk "BEGIN { printf \"%.1f\", ($umc_cas_rd * 64 / 1073741824) / $secs }")
        umc_wr_gbps=$(awk "BEGIN { printf \"%.1f\", ($umc_cas_wr * 64 / 1073741824) / $secs }")
        umc_total_gbps=$(awk "BEGIN { printf \"%.1f\", ($umc_total * 64 / 1073741824) / $secs }")
        echo -e "  │  ${C}${B}⚡ UMC read:  ${umc_rd_gbps} GB/s${D}"
        echo -e "  │  ${C}${B}⚡ UMC write: ${umc_wr_gbps} GB/s${D}"
        echo -e "  │  ${C}${B}⚡ UMC total: ${umc_total_gbps} GB/s${D}"
    else
        echo -e "  │  ${C}${B}⚡ ${core_rd_gbps} GB/s core reads${D}${C} (demand: ${dmnd_gbps} GB/s)${D}"
    fi
    if [ "$bw_label" = "perf window" ]; then
        echo -e "  │  ${C}  over ${bw_time}ms measurement window${D}"
    else
        echo -e "  │  ${C}  over ${bw_time}s elapsed (includes non-measured time)${D}"
    fi
fi
echo -e "  └─────────────────────────────────────────────────┘"

# ── Dispatch Stall Analysis ──
cycles=${cycles_line:-0}

# backend stall items: name cycles
declare -a be_names=() be_vals=()
add_be() { be_names+=("$1"); be_vals+=("$2"); }
add_be "Int reg file"   "$stall_int_reg"
add_be "Load queue"     "$stall_load_q"
add_be "Store queue"    "$stall_store_q"
add_be "FP reg file"    "$stall_fp_reg"
add_be "FP scheduler"   "$stall_fp_sch"
add_be "FP flush"       "$stall_fp_flush"
add_be "Branch buffer"  "$stall_taken_br"
add_be "Retire queue"   "$stall_retire"
add_be "Int sched 0"    "$stall_int_sch0"
add_be "Int sched 1"    "$stall_int_sch1"
add_be "Int sched 2"    "$stall_int_sch2"
add_be "Int sched 3"    "$stall_int_sch3"
add_be "Bitcoin (0xAE/08)" "$stall_bitcoin"

# frontend stall items
declare -a fe_names=() fe_vals=()
add_fe() { fe_names+=("$1"); fe_vals+=("$2"); }
add_fe "No ops from FE" "$stall_frontend"
add_fe "Op queue empty" "$stall_opq_empty"
add_fe "SMT contention" "$stall_smt"

has_stalls=0
for v in "$stall_backend" "${be_vals[@]}" "${fe_vals[@]}"; do
    if [ "$v" -gt 0 ] 2>/dev/null; then has_stalls=1; break; fi
done

if [ "$has_stalls" -eq 1 ] && [ "$cycles" -gt 0 ]; then
echo ""
echo -e "  ${B}╔══════════════════════════════════════════════════╗${D}"
echo -e "  ${B}║              DISPATCH STALL ANALYSIS             ║${D}"
echo -e "  ${B}╚══════════════════════════════════════════════════╝${D}"
echo ""

# Backend stalls
be_total=0
for v in "${be_vals[@]}"; do be_total=$((be_total + v)); done
be_pct=$(pct $stall_backend $cycles)
echo -e "  ${B}Backend Stalls${D}         $(fmt $stall_backend) cycles (${be_pct}% of total)"
echo -e "  ┌──────────────────────────────────────────────────┐"
# collect items above 5% of backend, rest goes to other
be_other=$stall_backend
shown=0
for i in "${!be_names[@]}"; do
    v=${be_vals[$i]}
    if [ "$v" -gt 0 ] && [ "$stall_backend" -gt 0 ]; then
        p=$(pct $v $stall_backend)
        above=$(awk "BEGIN { print ($p >= 5.0) ? 1 : 0 }")
        if [ "$above" -eq 1 ]; then
            echo -e "  │  ${Y}$(printf '%-18s' "${be_names[$i]}") $(fmt $v) (${p}%)${D}"
            be_other=$((be_other - v))
            shown=1
        fi
    fi
done
if [ "$be_other" -gt 0 ] && [ "$stall_backend" -gt 0 ]; then
    p=$(pct $be_other $stall_backend)
    echo -e "  │  ${W}$(printf '%-18s' "Other") $(fmt $be_other) (${p}%)${D}"
fi
if [ "$shown" -eq 0 ] && [ "$be_other" -le 0 ]; then
    echo -e "  │  ${W}(no significant stalls)${D}"
fi
echo -e "  └──────────────────────────────────────────────────┘"

# Frontend stalls
fe_total=0
for v in "${fe_vals[@]}"; do fe_total=$((fe_total + v)); done
if [ "$fe_total" -gt 0 ]; then
fe_slot_pct=$(pct $stall_frontend $cycles)
echo ""
echo -e "  ${B}Frontend Stalls${D}"
echo -e "  ┌──────────────────────────────────────────────────┐"
for i in "${!fe_names[@]}"; do
    v=${fe_vals[$i]}
    if [ "$v" -gt 0 ]; then
        p=$(pct $v $cycles)
        echo -e "  │  ${C}$(printf '%-18s' "${fe_names[$i]}") $(fmt $v) (${p}% of cycles)${D}"
    fi
done
echo -e "  └──────────────────────────────────────────────────┘"
fi

fi
echo ""