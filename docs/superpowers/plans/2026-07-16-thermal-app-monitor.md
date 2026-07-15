# Thermal App Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local thermal-status dashboard and top-three CPU application list with safe quit actions and restrained alerts.

**Architecture:** Pure monitoring policy and presentation types live in `NoSleepOwlCore`; AppKit adapters sample running GUI applications and expose an immutable snapshot. The existing ten-second safety loop publishes snapshots to the control window and status menu while retaining the current critical-thermal shutdown behavior.

**Tech Stack:** Swift 6.2, AppKit, Foundation, IOKit, Darwin process APIs, UserNotifications, existing custom Swift test executable.

## Global Constraints

- Do not read or display exact sensor temperatures.
- Do not add administrator privileges, network requests, analytics, or persisted history.
- Sample every ten seconds and show at most three user applications.
- Mark an application sustained-high only after two consecutive samples above 80% CPU.
- Never force-quit an application.
- Keep the existing critical thermal exit from owl mode.

---

### Task 1: Monitoring policy and presentation

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlCore/ThermalMonitoring.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Produces: `ThermalPresentation`, `AppUsage`, and `HighUsageTracker.evaluate(_:)`.

- [ ] **Step 1: Write failing tests** for four thermal titles, top-three sorting, and two-sample 80% high-usage detection.
- [ ] **Step 2: Run** `swift run NoSleepOwlTests`; expect missing-type compilation failures.
- [ ] **Step 3: Implement** value types and a tracker keyed by process identifier. The tracker sorts descending, returns three entries, increments only values above 80, and removes counters for absent processes.
- [ ] **Step 4: Run** `swift run NoSleepOwlTests`; expect all tests to pass.
- [ ] **Step 5: Commit** tests and core implementation with `feat: add thermal monitoring policy`.

### Task 2: Running application CPU sampler

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/ApplicationUsageSampler.swift`
- Modify: `NoSleepOwl/Package.swift`

**Interfaces:**
- Consumes: `AppUsage` from Task 1.
- Produces: `ApplicationUsageSampler.sample() -> [MonitoredApplication]` and `MonitoredApplication` metadata with PID, bundle identifier, name, icon, CPU percentage, and `canTerminate`.

- [ ] **Step 1: Add a small sampler fixture test target input** that validates CPU delta calculation `(newTime - oldTime) / elapsed * 100` and clamping at zero.
- [ ] **Step 2: Run** `swift run NoSleepOwlTests`; expect the missing calculator failure.
- [ ] **Step 3: Implement** a pure `CPUUsageCalculator` in core and an AppKit sampler using `NSWorkspace.shared.runningApplications`, `proc_pid_rusage`, and per-PID previous samples.
- [ ] **Step 4: Filter** the current app and applications without a localized name; retain system apps for display but set `canTerminate` false for protected bundle IDs and names.
- [ ] **Step 5: Run** tests and `swift build --product NoSleepOwlApp`; expect success.
- [ ] **Step 6: Commit** with `feat: sample running application CPU usage`.

### Task 3: Monitor orchestration and notifications

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalAppMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/SafetyMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

**Interfaces:**
- Produces: `ThermalAppSnapshot`, `ThermalAppMonitor.latestSnapshot`, `onChange`, and `terminate(pid:)`.
- Consumes: current thermal state, Task 1 tracker, Task 2 sampler.

- [ ] **Step 1: Write failing tests** for ten-minute serious-notification cooldown and reset behavior.
- [ ] **Step 2: Run** tests; expect missing notification-gate failure.
- [ ] **Step 3: Implement** `ThermalNotificationGate.shouldNotify(state:now:)` in core.
- [ ] **Step 4: Implement** the monitor, ten-second timer, snapshot publication, local notification request, and normal `NSRunningApplication.terminate()` action.
- [ ] **Step 5: Refactor** `SafetyMonitor` so thermal safety evaluates continuously while power-policy exits remain conditional on owl mode.
- [ ] **Step 6: Run** tests and app build; expect success.
- [ ] **Step 7: Commit** with `feat: monitor thermal state and hot applications`.

### Task 4: Control window and status menu UI

**Files:**
- Create: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalStatusView.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/StatusItemController.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/AppDelegate.swift`

**Interfaces:**
- Consumes: `ThermalAppMonitor.latestSnapshot`, `onChange`, and `terminate(pid:)`.
- Produces: a control-window status card and one disabled status-summary menu item.

- [ ] **Step 1: Build** a focused AppKit view with state title, semantic color, three application rows, icons, CPU values, sustained-high emphasis, and conditional quit buttons.
- [ ] **Step 2: Insert** the card into the existing control-window stack and increase the window height only as needed.
- [ ] **Step 3: Add** `电脑状态：<state> · <top app> <cpu>%` to the right-click menu, falling back to `正在获取应用占用`.
- [ ] **Step 4: Wire** monitor changes to refresh both UI surfaces on the main actor.
- [ ] **Step 5: Run** `swift run NoSleepOwlTests`, `swift build --product NoSleepOwlApp`, and the release build script.
- [ ] **Step 6: Install** the app, verify the card and menu visually, exercise a safe quit request against a disposable app, and confirm critical-state behavior remains covered by tests.
- [ ] **Step 7: Commit** with `feat: show thermal status and top applications`.
