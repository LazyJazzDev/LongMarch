#!/bin/sh
set -eu

# Override for another macOS DevEco installation without changing the project.
DEVECO_STUDIO_DIR="${DEVECO_STUDIO_DIR:-/Applications/DevEco-Studio.app/Contents}"
export NODE_HOME="$DEVECO_STUDIO_DIR/tools/node"
export JAVA_HOME="$DEVECO_STUDIO_DIR/jbr/Contents/Home"
export DEVECO_SDK_HOME="${DEVECO_SDK_HOME:-$DEVECO_STUDIO_DIR/sdk}"
export PATH="$NODE_HOME/bin:$PATH"
cd "$(dirname "$0")"
if [ ! -f entry/src/main/resources/rawfile/Resources/manifest.json ]; then
  echo "Prepare bundled resources first; see README.md." >&2
  exit 1
fi
"$DEVECO_STUDIO_DIR/tools/ohpm/bin/ohpm" install --all
exec "$DEVECO_STUDIO_DIR/tools/hvigor/bin/hvigorw" --mode module \
  -p product=default -p module=entry@default -p buildMode=debug \
  assembleHap --no-daemon "$@"
