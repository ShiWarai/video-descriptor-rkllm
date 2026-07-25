#!/usr/bin/env bash
# Pin RK3588 clocks for stable VLM inference (based on Rockchip rknn-llm fix_freq_rk3588.sh).
# Picks the **maximum** from each device's available_frequencies (Orange Pi big cores: 2256 MHz, etc.).
#
# Default: NPU + CPU + LPDDR4 (DMC) + Mali GPU at max. Set VLM_FIX_GPU_FREQ=0 to skip GPU.
# Requires write access to host sysfs (docker: privileged + /sys mounted).

set -euo pipefail

log() { echo "fix_freq: $*"; }
warn() { echo "fix_freq: warning: $*" >&2; }

# Largest numeric token from a space-separated frequency list (Hz).
max_from_file() {
    tr ' ' '\n' <"$1" | sort -n | tail -1
}

writable() {
    [[ -e "$1" ]] && [[ -w "$1" ]]
}

disable_cpu_deep_idle() {
    local cpu id
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        id="${cpu##*/cpu}"
        local state="${cpu}/cpuidle/state1/disable"
        if writable "$state"; then
            echo 1 >"$state" || warn "cpuidle disable failed for cpu${id}"
        fi
    done
}

set_devfreq_max() {
    local dev=$1
    local label=${2:-$dev}
    local avail="${dev}/available_frequencies"
    local gov="${dev}/governor"

    [[ -f "$avail" ]] || return 0
    local max_hz
    max_hz=$(max_from_file "$avail")
    [[ -n "$max_hz" ]] || return 0

    if ! writable "$gov"; then
        warn "no write access to ${gov}"
        return 0
    fi

    echo userspace >"$gov"
    if [[ -f "${dev}/userspace/set_freq" ]]; then
        echo "$max_hz" >"${dev}/userspace/set_freq"
    else
        warn "missing userspace/set_freq for ${label}"
        return 0
    fi

    local cur="?"
    if [[ -f "${dev}/cur_freq" ]]; then
        cur=$(<"${dev}/cur_freq")
    fi
    log "${label} ${max_hz} Hz (cur ${cur})"
}

set_cpufreq_policy_max() {
    local policy=$1
    local avail="${policy}/scaling_available_frequencies"
    local gov="${policy}/scaling_governor"
    local speed="${policy}/scaling_setspeed"

    [[ -f "$avail" ]] || return 0
    local max_hz
    max_hz=$(max_from_file "$avail")
    [[ -n "$max_hz" ]] || return 0

    if ! writable "$gov" || ! writable "$speed"; then
        warn "no write access to ${policy}"
        return 0
    fi

    echo userspace >"$gov"
    echo "$max_hz" >"$speed"

    local cur="?"
    if [[ -f "${policy}/scaling_cur_freq" ]]; then
        cur=$(<"${policy}/scaling_cur_freq")
    fi
    log "${policy##*/} ${max_hz} Hz (cur ${cur})"
}

main() {
    log "applying RK3588 performance cpufreq (best effort)"

    disable_cpu_deep_idle

    set_devfreq_max /sys/class/devfreq/fdab0000.npu NPU

    local policy
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        [[ -d "$policy" ]] || continue
        set_cpufreq_policy_max "$policy"
    done

    # LPDDR4 bandwidth matters for LLM + vision on Orange Pi — keep DMC at max.
    set_devfreq_max /sys/class/devfreq/dmc DDR

    if [[ "${VLM_FIX_GPU_FREQ:-1}" != "0" ]]; then
        set_devfreq_max /sys/class/devfreq/fb000000.gpu GPU
    else
        log "GPU left on system governor (VLM_FIX_GPU_FREQ=0)"
    fi

    log "done"
}

main "$@"
