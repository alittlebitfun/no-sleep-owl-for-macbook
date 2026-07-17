#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
PLIST="$ROOT/Resources/Info.plist"
grep -q 'iconScale.*0.88' "$ROOT/scripts/generate-app-icon.swift"

[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST")" == "0.1.0" ]]
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$PLIST")" == "7" ]]
[[ "$(sips -g pixelWidth "$ROOT/Resources/AppIcon.png" | awk '/pixelWidth/{print $2}')" == "1024" ]]
[[ "$(sips -g pixelHeight "$ROOT/Resources/AppIcon.png" | awk '/pixelHeight/{print $2}')" == "1024" ]]
iconutil --convert iconset --output "${TMPDIR:-/tmp}/NoSleepOwl-validation.iconset" "$ROOT/Resources/AppIcon.icns"
