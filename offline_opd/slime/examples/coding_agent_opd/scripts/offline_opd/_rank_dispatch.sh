#!/bin/bash
set -euo pipefail

rank="${OMPI_COMM_WORLD_RANK:?must run under mpirun}"
IFS=',' read -ra configured_hosts <<< "${PADDLE_TRAINERS:?PADDLE_TRAINERS is not set}"
expected_ip="${configured_hosts[rank]:?PADDLE_TRAINERS has no host for rank ${rank}}"
actual_ip="$(hostname -i | awk '{print $1}')"
if [[ "${actual_ip}" != "${expected_ip}" ]]; then
   echo "offline OPD rank ${rank} host mismatch: mpirun placed it on ${actual_ip}, but PADDLE_TRAINERS expects ${expected_ip}" >&2
   exit 1
fi

export PADDLE_TRAINER_ID="${rank}"
exec bash "${OFFLINE_OPD_RUN_SCRIPT:?OFFLINE_OPD_RUN_SCRIPT is not set}"
