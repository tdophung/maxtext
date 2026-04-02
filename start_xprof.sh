#!/bin/bash
# Launch xprof for each profile run in xprof_profiles/, each on a distinct port.
# Auto-discovered runs are sorted by profile capture timestamp (chronological),
# so port assignment matches the order the profiles were collected.
#
# Usage:
#   bash start_xprof.sh              # auto-discover all runs
#   bash start_xprof.sh te_impl_ring # only specific run(s)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/xprof_profiles_deepseek_1gpu}"
BASE_PORT="${BASE_PORT:-8791}"

# Return the earliest profile timestamp dir name for a tensorboard/ path.
_profile_timestamp() {
    ls -1 "$1/plugins/profile/" 2>/dev/null | sort | head -1
}

if [[ $# -gt 0 ]]; then
    dirs=()
    for name in "$@"; do
        d="${PROFILE_DIR}/${name}/tensorboard/"
        if [[ -d "$d" ]]; then
            dirs+=("$d")
        else
            echo "WARNING: ${d} not found, skipping."
        fi
    done
else
    # Collect tensorboard dirs sorted by profile capture timestamp so that
    # port assignment matches the chronological profiling order.
    unsorted=()
    for d in "${PROFILE_DIR}"/*/tensorboard/; do
        [[ -d "$d" ]] && unsorted+=("$d")
    done
    dirs=()
    while IFS= read -r line; do
        dirs+=("${line#* }")
    done < <(
        for d in "${unsorted[@]}"; do
            ts=$(_profile_timestamp "$d")
            echo "${ts:-zzz} ${d}"
        done | sort
    )
fi

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No profile tensorboard dirs found in ${PROFILE_DIR}/"
    exit 1
fi

MAX_PORT=$(( BASE_PORT + ${#dirs[@]} - 1 ))

# Clean up any leftover xprof processes / ports before starting.
export BASE_PORT MAX_PORT
bash "${SCRIPT_DIR}/kill_xprof.sh"

echo "Starting xprof servers..."
echo ""

port=${BASE_PORT}
for d in "${dirs[@]}"; do
    label=$(basename "$(dirname "$d")")
    echo "  ${label}  ->  http://localhost:${port}/"
    xprof --port ${port} "$d" &
    port=$((port + 1))
done

echo ""
echo "SSH tunnel (copy-paste to your local machine):"
tunnel_args=""
port=${BASE_PORT}
for d in "${dirs[@]}"; do
    tunnel_args+=" -L ${port}:localhost:${port}"
    port=$((port + 1))
done
echo "  ssh${tunnel_args} ptyche"
echo ""
echo "Press Ctrl+C to stop all xprof servers."
wait
