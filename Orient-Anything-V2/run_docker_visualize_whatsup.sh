#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: official visualization runs in the same host Conda
# environment used by the COCO example; Docker is not required.
exec "$(realpath "$(dirname "${BASH_SOURCE[0]}")")/run_visualize_whatsup.sh" "$@"
