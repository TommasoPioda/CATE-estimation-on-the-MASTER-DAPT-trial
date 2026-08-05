#!/usr/bin/env bash
#
# server_status.sh - Pretty server load/usage report
#
# Shows CPU, memory, disk, load average, uptime and top processes
# in a nicely formatted terminal output.
#
# Usage: ./server_status.sh

set -uo pipefail

# ---- Colors ----
RESET='\033[0m'
BOLD='\033[1m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BLUE='\033[34m'

WIDTH=56

hr() { printf '%*s\n' "$WIDTH" '' | tr ' ' '-'; }

title() {
    printf "${BOLD}${CYAN}%s${RESET}\n" "$1"
    hr
}

# Colorize a percentage value: green < 60, yellow < 85, red >= 85
color_pct() {
    local pct=$1
    pct=${pct%.*}
    if [ "$pct" -ge 85 ] 2>/dev/null; then
        echo -e "${RED}${pct}%${RESET}"
    elif [ "$pct" -ge 60 ] 2>/dev/null; then
        echo -e "${YELLOW}${pct}%${RESET}"
    else
        echo -e "${GREEN}${pct}%${RESET}"
    fi
}

bar() {
    local pct=$1
    pct=${pct%.*}
    [ -z "$pct" ] && pct=0
    local filled=$(( pct * 30 / 100 ))
    [ "$filled" -gt 30 ] && filled=30
    local empty=$(( 30 - filled ))
    local color=$GREEN
    if [ "$pct" -ge 85 ] 2>/dev/null; then color=$RED
    elif [ "$pct" -ge 60 ] 2>/dev/null; then color=$YELLOW
    fi
    printf "${color}"
    printf '%0.s#' $(seq 1 $filled) 2>/dev/null
    printf "${RESET}"
    printf '%0.s.' $(seq 1 $empty) 2>/dev/null
}

echo
printf "${BOLD}${BLUE}"
printf '=%.0s' $(seq 1 $WIDTH)
printf "${RESET}\n"
printf "${BOLD}${BLUE}  SERVER STATUS REPORT${RESET}\n"
printf "  %s | %s\n" "$(hostname)" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "${BOLD}${BLUE}"
printf '=%.0s' $(seq 1 $WIDTH)
printf "${RESET}\n\n"

# ---- Uptime & Load ----
title "UPTIME & LOAD AVERAGE"
UPTIME=$(uptime -p 2>/dev/null || uptime)
LOAD=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^ *//')
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "?")
printf "Uptime      : %s\n" "$UPTIME"
printf "Load avg    : %s (cores: %s)\n" "$LOAD" "$NCPU"
echo

# ---- CPU ----
title "CPU USAGE"
if command -v mpstat >/dev/null 2>&1; then
    CPU_IDLE=$(mpstat 1 1 | awk '/Average/ {print $NF}')
    CPU_USED=$(awk -v idle="$CPU_IDLE" 'BEGIN{printf "%.0f", 100-idle}')
else
    CPU_USED=$(top -bn1 2>/dev/null | grep -i "Cpu(s)" | awk -F',' '{print $4}' | awk '{print $1}' | sed 's/%//;s/id//')
    CPU_USED=$(awk -v idle="${CPU_USED:-0}" 'BEGIN{printf "%.0f", 100-idle}')
fi
printf "Usage       : [%s] %s\n" "$(bar "${CPU_USED:-0}")" "$(color_pct "${CPU_USED:-0}")"
echo

# ---- Memory ----
title "MEMORY USAGE"
if command -v free >/dev/null 2>&1; then
    read -r MEM_TOTAL MEM_USED MEM_FREE <<< "$(free -m | awk '/Mem:/ {print $2, $3, $4}')"
    MEM_PCT=$(awk -v u="$MEM_USED" -v t="$MEM_TOTAL" 'BEGIN{printf "%.0f", (u/t)*100}')
    printf "Usage       : [%s] %s\n" "$(bar "$MEM_PCT")" "$(color_pct "$MEM_PCT")"
    printf "Used/Total  : %s MB / %s MB\n" "$MEM_USED" "$MEM_TOTAL"
else
    echo "free command not available"
fi
echo

# ---- Disk ----
title "DISK USAGE"
df -h --output=target,pcent,used,size -x tmpfs -x devtmpfs 2>/dev/null | tail -n +2 | while read -r mount pct used size; do
    p=${pct%\%}
    printf "%-20s [%s] %-4s (%s/%s)\n" "$mount" "$(bar "$p")" "$(color_pct "$p")" "$used" "$size"
done
echo

# ---- Top processes ----
title "TOP 5 PROCESSES (CPU)"
ps -eo pid,comm,%cpu,%mem --sort=-%cpu 2>/dev/null | head -n 6 | awk 'NR==1{printf "%-8s %-20s %-6s %-6s\n",$1,$2,$3,$4; next} {printf "%-8s %-20s %-6s %-6s\n",$1,$2,$3,$4}'
echo

printf "${BOLD}${BLUE}"
printf '=%.0s' $(seq 1 $WIDTH)
printf "${RESET}\n"