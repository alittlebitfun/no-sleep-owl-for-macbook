#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP="$ROOT/dist/不休眠猫头鹰.app"

cd "$ROOT"
swift build -c release --arch arm64
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Library/LaunchDaemons"
cp "$ROOT/.build/arm64-apple-macosx/release/NoSleepOwlApp" "$APP/Contents/MacOS/NoSleepOwlApp"
cp "$ROOT/.build/arm64-apple-macosx/release/NoSleepOwlHelper" "$APP/Contents/Resources/NoSleepOwlHelper"
cp "$ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cp "$ROOT/Resources/com.shiying.NoSleepOwl.helper.plist" "$APP/Contents/Library/LaunchDaemons/com.shiying.NoSleepOwl.helper.plist"
codesign --force --sign - --identifier com.shiying.NoSleepOwl.helper "$APP/Contents/Resources/NoSleepOwlHelper"
codesign --force --deep --sign - "$APP"
echo "$APP"
