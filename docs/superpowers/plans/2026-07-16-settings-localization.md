# Settings and Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native settings window with immediate Chinese/English switching and independent visibility controls for thermal status and high-CPU applications.

**Architecture:** A core preference model owns persisted values and display policies. A strongly typed string catalog supplies all user-facing copy. AppKit controllers consume one shared preference store, and the thermal monitor skips process sampling when application usage is hidden while retaining thermal safety monitoring.

**Tech Stack:** Swift 6, AppKit, Foundation `UserDefaults`, Swift Package Manager, existing executable test harness.

## Global Constraints

- Support macOS 15 or newer on Apple silicon arm64.
- Default to Simplified Chinese with thermal status and high-usage applications visible.
- Language and visibility changes take effect immediately and persist across relaunches.
- Hiding thermal information never disables thermal sampling, notifications, or automatic safety protection.
- Hiding application usage pauses process CPU sampling.
- Keep the installed application identity and bundle name as “不休眠猫头鹰”.
- User-facing copy must use direct affirmative wording.

---

### Task 1: Preference model and display policy

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/AppPreferences.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Produces: `AppLanguage`, `AppPreferenceSnapshot`, `AppPreferences`, `MonitoringDisplayMode`, and `MonitoringDisplayPolicy.mode(thermal:applications:)`.
- Consumes: Foundation `UserDefaults`.

- [ ] **Step 1: Write failing tests for defaults, persistence, fallback, and four display combinations**

```swift
test("preferences default to Chinese with both monitors visible") {
    let defaults = isolatedDefaults()
    let preferences = AppPreferences(defaults: defaults)
    try expect(preferences.snapshot == AppPreferenceSnapshot(language: .zhHans, showsThermalStatus: true, showsHighUsageApps: true), "wrong defaults")
}

test("monitoring display policy covers all combinations") {
    try expect(MonitoringDisplayPolicy.mode(thermal: true, applications: true) == .full, "full mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: true, applications: false) == .thermalOnly, "thermal mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: false, applications: true) == .applicationsOnly, "applications mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: false, applications: false) == .hidden, "hidden mode")
}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: compilation fails because `AppPreferences` and `MonitoringDisplayPolicy` do not exist.

- [ ] **Step 3: Implement the preference model**

```swift
public enum AppLanguage: String, Sendable, CaseIterable { case zhHans, en }
public struct AppPreferenceSnapshot: Sendable, Equatable {
    public let language: AppLanguage
    public let showsThermalStatus: Bool
    public let showsHighUsageApps: Bool
}
public final class AppPreferences {
    public private(set) var snapshot: AppPreferenceSnapshot
    public var onChange: (() -> Void)?
    public init(defaults: UserDefaults = .standard)
    public func setLanguage(_ value: AppLanguage)
    public func setShowsThermalStatus(_ value: Bool)
    public func setShowsHighUsageApps(_ value: Bool)
}
public enum MonitoringDisplayMode { case full, thermalOnly, applicationsOnly, hidden }
```

Unknown stored language strings resolve to `.zhHans`. Setters write to the injected defaults and call `onChange` once only when the value changes.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add NoSleepOwl/Sources/NoSleepOwlCore/AppPreferences.swift NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift
git commit -m "feat: add persistent app preferences"
```

### Task 2: Typed Chinese and English strings

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/AppStrings.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlCore/BirdPresentation.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlCore/ThermalMonitoring.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Consumes: `AppLanguage`, `OwlMode`, `OwlThermalState`, and dynamic names/values.
- Produces: `AppStrings(language:)`, localized `BirdPresentation(mode:language:)`, and `ThermalPresentation(state:language:)`.

- [ ] **Step 1: Write failing bilingual copy tests**

```swift
test("English strings cover settings and monitoring") {
    let strings = AppStrings(language: .en)
    try expect(strings.settingsMenuTitle == "Settings…", "settings title")
    try expect(strings.showThermalStatus == "Show thermal status", "thermal setting")
    try expect(BirdPresentation(mode: .owl, language: .en).statusTitle == "Owl is keeping watch", "owl title")
    try expect(ThermalPresentation(state: .serious, language: .en).title == "Serious", "thermal title")
}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: compilation fails because `AppStrings` and language-aware initializers are missing.

- [ ] **Step 3: Implement all strings required by AppKit controllers**

`AppStrings` includes window titles, settings labels, menu titles, monitoring placeholders, notification copy, helper states, safety messages, errors, and dynamic formatter methods. Existing initializers retain Chinese defaults so older tests and callers remain source compatible.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: all tests pass in both languages.

- [ ] **Step 5: Commit**

```bash
git add NoSleepOwl/Sources/NoSleepOwlCore/AppStrings.swift NoSleepOwl/Sources/NoSleepOwlCore/BirdPresentation.swift NoSleepOwl/Sources/NoSleepOwlCore/ThermalMonitoring.swift NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift
git commit -m "feat: add Chinese and English copy"
```

### Task 3: Native settings window

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/SettingsWindowController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Consumes: shared `AppPreferences` and `AppStrings`.
- Produces: `SettingsWindowController.show()`, `refresh()`, and a right-click menu “Settings…” action.

- [ ] **Step 1: Write a failing menu composition policy test**

```swift
test("settings entry is always available") {
    try expect(StatusMenuPolicy.includesSettings == true, "settings must be reachable")
}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: compilation fails because `StatusMenuPolicy` is missing.

- [ ] **Step 3: Build the settings controller and wire the menu action**

The window is 420×250 points with a language popup and two checkboxes. It is retained, reused, centered on first display, and activates the app when shown. Each control writes through the shared preference store. `refresh()` updates titles, selected language, and checkbox state without triggering actions.

- [ ] **Step 4: Run tests and build the application**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests && swift build`
Expected: all tests pass and all AppKit sources compile.

- [ ] **Step 5: Commit**

```bash
git add NoSleepOwl/Sources/NoSleepOwlApp/SettingsWindowController.swift NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift NoSleepOwl/Sources/NoSleepOwlCore/StatusBarInteraction.swift NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift
git commit -m "feat: add native settings window"
```

### Task 4: Apply visibility settings and sampling optimization

**Files:**
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalAppMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalStatusView.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Consumes: `AppPreferenceSnapshot`, `MonitoringDisplayMode`, and `AppStrings`.
- Produces: `ThermalAppMonitor.setShowsApplicationUsage(_:)` and `ThermalStatusView.update(snapshot:mode:strings:)`.

- [ ] **Step 1: Write failing sampling and menu visibility policy tests**

```swift
test("application sampling follows its visibility preference") {
    try expect(MonitoringSamplingPolicy.samplesApplications(showsHighUsageApps: true), "visible list samples")
    try expect(!MonitoringSamplingPolicy.samplesApplications(showsHighUsageApps: false), "hidden list pauses")
}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: compilation fails because `MonitoringSamplingPolicy` is missing.

- [ ] **Step 3: Implement four display modes and sampling control**

When application usage is disabled, the monitor skips `ApplicationUsageSampler.sample`, publishes an empty application array, and retains the real thermal state. Re-enabling triggers an immediate evaluation. The status card uses full, thermal-only, applications-only, or hidden mode; the menu applies the same policy.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: all policy and regression tests pass.

- [ ] **Step 5: Commit**

```bash
git add NoSleepOwl/Sources/NoSleepOwlApp/ThermalAppMonitor.swift NoSleepOwl/Sources/NoSleepOwlApp/ThermalStatusView.swift NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift
git commit -m "feat: apply monitoring visibility settings"
```

### Task 5: Localize remaining application surfaces

**Files:**
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalStatusView.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalAppMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/SafetyMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/PrivilegedSleepController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlCore/OwlModeStore.swift`

**Interfaces:**
- Consumes: current `AppLanguage` and `AppStrings` through the shared preference snapshot.
- Produces: immediate bilingual refresh across control window, menu, settings, notifications, helper status, safety messages, and errors.

- [ ] **Step 1: Add tests for every dynamic copy category**

```swift
test("dynamic application copy is bilingual") {
    let zh = AppStrings(language: .zhHans)
    let en = AppStrings(language: .en)
    try expect(zh.duration(65).contains("01:05"), "Chinese duration")
    try expect(en.duration(65).contains("01:05"), "English duration")
    try expect(zh.loginAtStartup == "登录时自动启动", "Chinese login label")
    try expect(en.loginAtStartup == "Launch at Login", "English login label")
    try expect(zh.quitApplication.contains("退出"), "Chinese quit label")
    try expect(en.quitApplication == "Quit No Sleep Owl", "English quit label")
    try expect(en.helperApproved.contains("approved"), "English helper state")
    try expect(en.lowBatteryWarning.contains("20%"), "English battery warning")
    try expect(en.thermalWarning.lowercased().contains("thermal"), "English thermal warning")
}
```

- [ ] **Step 2: Run tests and verify RED for untranslated accessors**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests`
Expected: tests fail for missing localized accessors.

- [ ] **Step 3: Replace user-visible literals with `AppStrings` accessors**

Keep low-level technical error descriptions available for diagnostics, but wrap all displayed messages with localized copy. Existing mode and safety state machines remain behaviorally unchanged.

- [ ] **Step 4: Run full tests and scan for remaining Chinese literals in AppKit views**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests && rg -n '"[^"\\]*[一-龥]' Sources/NoSleepOwlApp`
Expected: tests pass; remaining Chinese literals are application identity strings or low-level diagnostic fallback values only.

- [ ] **Step 5: Commit**

```bash
git add NoSleepOwl/Sources NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift
git commit -m "feat: localize application surfaces"
```

### Task 6: Build, install, and visual acceptance

**Files:**
- Verify: `NoSleepOwl/dist/不休眠猫头鹰.app`
- Verify: `/Applications/不休眠猫头鹰.app`

**Interfaces:**
- Consumes: completed application.
- Produces: signed installed arm64 app at `/Applications/不休眠猫头鹰.app`.

- [ ] **Step 1: Run final automated verification**

Run: `cd NoSleepOwl && swift run NoSleepOwlTests && ./scripts/build-app.sh && codesign --verify --deep --strict dist/不休眠猫头鹰.app`
Expected: zero test failures, successful release build, valid signature.

- [ ] **Step 2: Install and launch**

```bash
pkill -x NoSleepOwlApp 2>/dev/null || true
rm -rf /Applications/不休眠猫头鹰.app
ditto NoSleepOwl/dist/不休眠猫头鹰.app /Applications/不休眠猫头鹰.app
open -a /Applications/不休眠猫头鹰.app
```

- [ ] **Step 3: Perform actual UI acceptance**

Verify through the running application: settings menu entry, single reusable settings window, immediate English switch across all open windows, thermal-only mode, applications-only mode, fully hidden card, restored full mode, and persisted settings after relaunch.

- [ ] **Step 4: Confirm resource and menu-bar behavior**

Confirm one native status item, zero floating overlays, and low idle CPU when application sampling is disabled.

- [ ] **Step 5: Commit any scoped verification corrections**

```bash
git add NoSleepOwl/Sources NoSleepOwl/Tests
git commit -m "fix: polish settings acceptance"
```
