# Top Seven Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show seven ranked applications and reduce background monitoring work when the control window is closed and thermal state is normal.

**Architecture:** Extend the pure core ranking and sampling-interval policy first. Then add PID path caching in the sampler and a fixed-height scroll view in AppKit.

**Tech Stack:** Swift 6.2, AppKit, Darwin libproc, existing custom Swift tests.

## Global Constraints

- Show at most seven applications.
- Sample every ten seconds when the window is visible or thermal state is elevated.
- Sample every twenty seconds only when the window is closed and thermal state is normal.
- Keep right-click menu summary to one application.
- Preserve current safety and notification behavior.

---

### Task 1: Ranking and interval policy

**Files:**
- Modify: `NoSleepOwl/Sources/NoSleepOwlCore/ThermalMonitoring.swift`
- Modify: `NoSleepOwl/Tests/NoSleepOwlCoreTests/main.swift`

- [ ] Write failing tests proving seven-item truncation and the ten/twenty-second interval matrix.
- [ ] Run `swift run NoSleepOwlTests` and confirm missing behavior.
- [ ] Change `HighUsageTracker` to return seven items and add `MonitoringIntervalPolicy.interval(windowVisible:thermalState:)`.
- [ ] Run the full tests and confirm success.

### Task 2: Cache, scrolling UI, and installation

**Files:**
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ApplicationUsageSampler.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalAppMonitor.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ThermalStatusView.swift`
- Modify: `NoSleepOwl/Sources/NoSleepOwlApp/ControlWindowController.swift`

- [ ] Cache executable paths by PID and remove cache entries for PIDs no longer returned by `proc_listallpids`.
- [ ] Replace the repeating timer with a rescheduled one-shot timer using the core interval policy.
- [ ] Notify the monitor when the control window shows or closes.
- [ ] Put application rows in a fixed-height `NSScrollView` and retain seven rows.
- [ ] Run tests, build the release app, install it, and visually verify seven ranked applications and scrolling.
- [ ] Commit with `feat: show top seven apps efficiently`.
