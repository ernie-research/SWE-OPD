#!/bin/bash
set -euo pipefail

threshold_mib="${GPU_IDLE_THRESHOLD_MIB:-2048}"
max_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN { max=0 } { if ($1 > max) max=$1 } END { print max }')"
printf 'host=%s max_gpu_memory_mib=%s threshold_mib=%s\n' "$(hostname)" "${max_used}" "${threshold_mib}"
(( max_used <= threshold_mib ))
