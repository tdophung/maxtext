#!/bin/bash
# Kill all running xprof processes owned by the current user,
# then force-free the ports in case any leaked sockets remain.

BASE_PORT="${BASE_PORT:-8791}"
MAX_PORT="${MAX_PORT:-8794}"

# --- Pass 1: kill by process name ---
# Exclude our own PID and parent PID so we don't kill start_xprof.sh or ourselves.
pids=$(pgrep -u "$(whoami)" -f 'xprof' | grep -v -w -e "$$" -e "$PPID" | tr '\n' ' ')
if [[ -n "${pids}" ]]; then
    echo "Killing xprof processes (by name):"
    ps -p ${pids} -o pid,args --no-headers 2>/dev/null
    kill -9 ${pids} 2>/dev/null
fi

# --- Pass 2: kill anything still holding the ports ---
port_pids=""
for port in $(seq "${BASE_PORT}" "${MAX_PORT}"); do
    holder=$(fuser "${port}/tcp" 2>/dev/null)
    if [[ -n "${holder}" ]]; then
        port_pids+=" ${holder}"
    fi
done

if [[ -n "${port_pids}" ]]; then
    echo "Killing leftover processes on ports ${BASE_PORT}-${MAX_PORT}:"
    ps -p ${port_pids} -o pid,args --no-headers 2>/dev/null
    kill -9 ${port_pids} 2>/dev/null
fi

if [[ -z "${pids}" && -z "${port_pids}" ]]; then
    echo "No xprof processes found."
else
    echo "Done."
fi
