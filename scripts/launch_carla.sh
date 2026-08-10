#!/usr/bin/env bash
set -euo pipefail

mode="${1:-offscreen}"
carla_root="${CARLA_ROOT:-.}"

cd "${carla_root}"

case "${mode}" in
  offscreen)
    exec ./CarlaUE4.sh -RenderOffScreen -quality_level=Low-prefernvidia
    ;;
  window)
    exec ./CarlaUE4.sh -quality_level=Low-prefernvidia
    ;;
  *)
    echo "Usage: CARLA_ROOT=/path/to/CARLA $0 [offscreen|window]" >&2
    exit 2
    ;;
esac
