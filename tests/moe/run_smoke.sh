#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: run_smoke.sh binary ag|rs route output_directory [world_size]}
operator=${2:?missing operator}
route=${3:?missing route}
output_directory=${4:?missing output directory}
world_size=${5:-8}
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$script_directory/../support/run_multirank.py" \
    "$binary" "$operator" "$route" "$output_directory" --world-size "$world_size" --timeout 180 --gtest
