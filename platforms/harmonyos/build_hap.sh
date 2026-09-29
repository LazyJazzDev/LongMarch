#!/bin/sh
set -eu

# Override for another macOS DevEco installation without changing the project.
DEVECO_STUDIO_DIR="${DEVECO_STUDIO_DIR:-/Applications/DevEco-Studio.app/Contents}"
export NODE_HOME="$DEVECO_STUDIO_DIR/tools/node"
export JAVA_HOME="$DEVECO_STUDIO_DIR/jbr/Contents/Home"
export DEVECO_SDK_HOME="${DEVECO_SDK_HOME:-$DEVECO_STUDIO_DIR/sdk}"
export PATH="$NODE_HOME/bin:$PATH"
# PRODUCT=gameoflife builds the standalone Game of Life; BUILD_MODE=release with
# TASK=assembleApp builds the signed .app package for AppGallery.
PRODUCT="${PRODUCT:-default}"
BUILD_MODE="${BUILD_MODE:-debug}"
TASK="${TASK:-assembleHap}"
cd "$(dirname "$0")"
case "$PRODUCT" in
  gameoflife) RESOURCES=entry/src/gol/resources/rawfile/Resources ;;
  *) RESOURCES=entry/src/demos/resources/rawfile/Resources ;;
esac
if [ ! -f "$RESOURCES/manifest.json" ]; then
  echo "Prepare bundled resources in $RESOURCES first; see README.md." >&2
  exit 1
fi
"$DEVECO_STUDIO_DIR/tools/ohpm/bin/ohpm" install --all
if [ "$TASK" = assembleApp ]; then
  exec "$DEVECO_STUDIO_DIR/tools/hvigor/bin/hvigorw" --mode project \
    -p product="$PRODUCT" -p buildMode="$BUILD_MODE" assembleApp --no-daemon "$@"
fi
exec "$DEVECO_STUDIO_DIR/tools/hvigor/bin/hvigorw" --mode module \
  -p product="$PRODUCT" -p module=entry@"$PRODUCT" -p buildMode="$BUILD_MODE" \
  "$TASK" --no-daemon "$@"
