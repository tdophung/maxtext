#!/bin/bash
# Launch xprof for each profile run in xprof_profiles/, each on a distinct port.
# Usage:
#   bash start_xprof.sh              # auto-discover all runs
#   bash start_xprof.sh te_impl_ring # only specific run(s)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-${SCRIPT_DIR}/xprof_profiles}"
BASE_PORT="${BASE_PORT:-8791}"

# Count how many profile dirs we'll use so we can clean up the right port range.
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
    dirs=()
    for d in "${PROFILE_DIR}"/*/tensorboard/; do
        [[ -d "$d" ]] && dirs+=("$d")
    done
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
