# Display Location Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live settings for showing the menu bar item and Dock icon, with persisted preferences and no loss of the running background service.

**Architecture:** Store two booleans in `AppPreferences`; let `SettingsWindowController` edit them; use a small `DisplayLocationController` owned by `AppDelegate` to create/destroy the status item and switch activation policy. Keep the existing `StatusItemController` as the menu bar item implementation and preserve all sleep/monitoring behavior.

**Tech Stack:** Swift, AppKit, Swift Package Manager, UserDefaults, XCTest-style executable tests.

## Global Constraints

- Default values are status bar on and Dock off.
- Changes apply immediately without relaunch.
- Both options off keeps the process alive and does not terminate sleep prevention or monitoring.
- Existing language, thermal, application usage, menu, and reopen behavior remain intact.
- Apple silicon release build remains `arm64`.

---

### Task 1: Persist Display Preferences and Localized Settings Copy

**Files:**
- Modify: `Sources/NoSleepOwlCore/AppPreferences.swift`
- Modify: `Sources/NoSleepOwlCore/AppStrings.swift`
- Modify: `Sources/NoSleepOwlApp/SettingsWindowController.swift`
- Test: `Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Add `showsStatusBarIcon` and `showsDockIcon` to `AppPreferenceSnapshot` and `AppPreferences`.
- Add `setShowsStatusBarIcon(_:)` and `setShowsDockIcon(_:)`.

- [ ] Add failing tests for defaults, persistence, unknown values, and Chinese/English labels.
- [ ] Run `NO_SLEEP_OWL_ROOT="$PWD" swift run NoSleepOwlTests`; confirm the new symbols/tests fail.
- [ ] Implement the two persisted booleans, defaulting to `true` and `false`.
- [ ] Add localized strings and two checkboxes to the existing settings window.
- [ ] Wire checkbox actions to the new preference setters.
- [ ] Run the full test executable and confirm all tests pass.
- [ ] Commit with `feat: add display location preferences`.

### Task 2: Add Runtime Display Location Controller

**Files:**
- Create: `Sources/NoSleepOwlApp/DisplayLocationController.swift`
- Modify: `Sources/NoSleepOwlApp/AppDelegate.swift`
- Modify: `Sources/NoSleepOwlCore/StatusBarInteraction.swift`
- Test: `Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- `DisplayLocationController.apply(_ snapshot: AppPreferenceSnapshot)` updates `NSApplication.ActivationPolicy` and owns an optional `StatusItemController`.

- [ ] Add failing tests for all four combinations and the “both off keeps process alive” policy.
- [ ] Run the test executable and confirm failure from the missing policy/controller.
- [ ] Implement one-controller-only lifecycle: instantiate status item on enable, release it on disable, and call `NSApp.setActivationPolicy(.regular/.accessory)` for Dock visibility.
- [ ] Rewire `AppDelegate` callbacks through the controller and keep window/settings closures valid after recreation.
- [ ] Apply the initial snapshot after window controllers are created and apply changes from `preferences.onChange`.
- [ ] Run full tests and verify no duplicate status item is created.
- [ ] Commit with `feat: switch display locations at runtime`.

### Task 3: Build, Install, and Verify

**Files:**
- Modify: `Resources/Info.plist` build number only.
- Generated: `Resources/AppIcon.icns` and `dist/不休眠猫头鹰.app` only through existing scripts.

- [ ] Increment `CFBundleVersion` by one.
- [ ] Run validation and all tests.
- [ ] Build with `zsh scripts/build-app.sh` and verify arm64, version, signature, and icon.
- [ ] Install to `/Applications`, launch, and verify settings window exposes both switches.
- [ ] Toggle status bar and Dock visibility in every combination, verifying immediate effect and process persistence.
- [ ] Verify reopening the app opens the control window.
- [ ] Commit with `chore: build display location settings release`.
