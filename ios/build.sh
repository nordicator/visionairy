#!/usr/bin/env bash
# Build the AR app and install it on the connected iPhone. The laptop's current Wi-Fi address is baked
# in as the server to connect to, so re-run this if the laptop's address changes.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${1:-$(ipconfig getifaddr en0)}"
xcodegen generate --quiet
xcodebuild -project VisionAiryAR.xcodeproj -scheme VisionAiryAR -configuration Debug \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates \
  VA_SERVER_HOST="$HOST" build -quiet
APP=build/Build/Products/Debug-iphoneos/VisionAiryAR.app
echo "Built $APP (server $HOST)"

# Whichever paired iPhone is reachable right now (the list also keeps phones that are offline)
DEVICE=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && !/unavailable/ {print $3; exit}')
if [ -z "$DEVICE" ]; then echo "No iPhone connected: plug it in (or join the same Wi-Fi) and re-run"; exit 1; fi
xcrun devicectl device install app --device "$DEVICE" "$APP"
xcrun devicectl device process launch --device "$DEVICE" com.cdx.visionairy.ar || true
