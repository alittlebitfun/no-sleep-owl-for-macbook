import Foundation
import NoSleepOwlCore

private var failures = 0

@MainActor
private func test(_ name: String, _ body: () throws -> Void) {
    do {
        try body()
        print("PASS \(name)")
    } catch {
        failures += 1
        print("FAIL \(name): \(error)")
    }
}

@MainActor
private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else { throw TestError.expectation(message) }
}

private func isolatedDefaults() -> UserDefaults {
    let suite = "NoSleepOwlTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defaults.removePersistentDomain(forName: suite)
    return defaults
}

test("preferences default to Chinese with both monitors visible") {
    let preferences = AppPreferences(defaults: isolatedDefaults())
    try expect(preferences.snapshot == AppPreferenceSnapshot(language: .zhHans, showsThermalStatus: true, showsHighUsageApps: true), "wrong defaults")
}

test("preferences persist and unknown language falls back to Chinese") {
    let defaults = isolatedDefaults()
    let preferences = AppPreferences(defaults: defaults)
    preferences.setLanguage(.en)
    preferences.setShowsThermalStatus(false)
    preferences.setShowsHighUsageApps(false)
    try expect(AppPreferences(defaults: defaults).snapshot == AppPreferenceSnapshot(language: .en, showsThermalStatus: false, showsHighUsageApps: false), "preferences did not persist")
    defaults.set("unknown", forKey: AppPreferences.Keys.language)
    try expect(AppPreferences(defaults: defaults).snapshot.language == .zhHans, "unknown language must fall back")
}

test("monitoring display policy covers all combinations") {
    try expect(MonitoringDisplayPolicy.mode(thermal: true, applications: true) == .full, "full mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: true, applications: false) == .thermalOnly, "thermal mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: false, applications: true) == .applicationsOnly, "applications mode")
    try expect(MonitoringDisplayPolicy.mode(thermal: false, applications: false) == .hidden, "hidden mode")
}

test("English strings cover settings and monitoring") {
    let strings = AppStrings(language: .en)
    try expect(strings.settingsMenuTitle == "Settings…", "settings title")
    try expect(strings.showThermalStatus == "Show thermal status", "thermal setting")
    try expect(BirdPresentation(mode: .owl, language: .en).statusTitle == "Owl is keeping watch", "owl title")
    try expect(ThermalPresentation(state: .serious, language: .en).title == "Serious", "thermal title")
}

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

test("starts in bird mode") {
    let store = OwlModeStore(controller: FakeSleepAssertionController())
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.startedAt == nil, "expected no start date")
}

test("toggle acquires before entering owl mode") {
    let controller = FakeSleepAssertionController(acquiredID: 42)
    let now = Date(timeIntervalSince1970: 100)
    let store = OwlModeStore(controller: controller, now: { now })
    store.toggle()
    try expect(store.mode == .owl, "expected owl mode")
    try expect(store.startedAt == now, "expected captured start date")
    try expect(controller.acquireCount == 1, "expected one acquire")
    try expect(store.errorMessage == nil, "expected no error")
}

test("acquire failure keeps bird mode") {
    let store = OwlModeStore(controller: FakeSleepAssertionController(acquireError: TestError.failed))
    store.toggle()
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.errorMessage != nil, "expected error message")
}

test("toggle releases before returning to bird mode") {
    let controller = FakeSleepAssertionController(acquiredID: 7)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.toggle()
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.startedAt == nil, "expected cleared start date")
    try expect(controller.releasedIDs == [7], "expected assertion 7 released")
}

test("release failure keeps owl mode") {
    let controller = FakeSleepAssertionController(acquiredID: 9, releaseError: TestError.failed)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.toggle()
    try expect(store.mode == .owl, "expected owl mode")
    try expect(store.errorMessage != nil, "expected error message")
}

test("shutdown releases active assertion") {
    let controller = FakeSleepAssertionController(acquiredID: 11)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.shutdown()
    try expect(controller.releasedIDs == [11], "expected assertion 11 released")
    try expect(store.mode == .bird, "expected bird mode")
}

test("IOKit assertion prevents only idle system sleep") {
    try expect(IOKitSleepAssertionController.assertionType == "PreventUserIdleSystemSleep", "wrong assertion type")
}

test("bird and owl presentations are distinct") {
    try expect(BirdPresentation(mode: .bird).statusTitle == "小鸟可以休息", "wrong bird title")
    try expect(BirdPresentation(mode: .owl).statusTitle == "猫头鹰正在守夜", "wrong owl title")
    try expect(BirdPresentation(mode: .bird).toggleTitle == "切换到猫头鹰模式", "wrong bird action")
    try expect(BirdPresentation(mode: .owl).toggleTitle == "切换到小鸟模式", "wrong owl action")
}

test("primary menu bar click opens the control window") {
    try expect(StatusBarInteraction.action(for: .primary) == .openControlWindow, "primary click must open window")
}

test("secondary menu bar click opens the context menu") {
    try expect(StatusBarInteraction.action(for: .secondary) == .showContextMenu, "secondary click must show menu")
}

test("menu bar icon uses only system managed placement") {
    try expect(StatusItemPlacementPolicy.usesFloatingOverlay == false, "floating overlays can cover the menu bar or drift near the Dock")
    try expect(StatusItemPlacementPolicy.persistsCustomPosition == false, "custom status item positions can restore off-screen")
}

test("settings entry is always available") {
    try expect(StatusMenuPolicy.includesSettings, "settings must be reachable")
}

test("application sampling follows its visibility preference") {
    try expect(MonitoringSamplingPolicy.samplesApplications(showsHighUsageApps: true), "visible list samples")
    try expect(!MonitoringSamplingPolicy.samplesApplications(showsHighUsageApps: false), "hidden list pauses")
}

test("duration formatter covers seconds minutes and hours") {
    try expect(OwlDurationFormatter.string(seconds: 0) == "00:00", "wrong zero duration")
    try expect(OwlDurationFormatter.string(seconds: 65) == "01:05", "wrong minute duration")
    try expect(OwlDurationFormatter.string(seconds: 3661) == "01:01:01", "wrong hour duration")
}

test("AC-only policy exits when power is unplugged") {
    var policy = SafetyPolicy()
    let decision = policy.evaluate(SafetySnapshot(policy: .acOnly, isOnACPower: false, batteryPercent: 80, thermalState: .nominal))
    try expect(decision == .exitOwl(reason: .powerDisconnected), "expected unplug exit")
}

test("battery policy warns once at twenty percent") {
    var policy = SafetyPolicy()
    let snapshot = SafetySnapshot(policy: .allowBattery, isOnACPower: false, batteryPercent: 20, thermalState: .nominal)
    try expect(policy.evaluate(snapshot) == .warn(.lowBattery), "expected low battery warning")
    try expect(policy.evaluate(snapshot) == .none, "warning must fire once")
}

test("battery policy exits at ten percent") {
    var policy = SafetyPolicy()
    let decision = policy.evaluate(SafetySnapshot(policy: .allowBattery, isOnACPower: false, batteryPercent: 10, thermalState: .nominal))
    try expect(decision == .exitOwl(reason: .criticalBattery), "expected critical battery exit")
}

test("thermal policy warns at serious and exits at critical") {
    var policy = SafetyPolicy()
    let serious = SafetySnapshot(policy: .allowBattery, isOnACPower: true, batteryPercent: 100, thermalState: .serious)
    let critical = SafetySnapshot(policy: .allowBattery, isOnACPower: true, batteryPercent: 100, thermalState: .critical)
    try expect(policy.evaluate(serious) == .warn(.thermalSerious), "expected thermal warning")
    try expect(policy.evaluate(serious) == .none, "thermal warning must fire once")
    try expect(policy.evaluate(critical) == .exitOwl(reason: .thermalCritical), "expected thermal exit")
}

test("thermal presentation maps all system states") {
    try expect(ThermalPresentation(state: .nominal).title == "正常", "wrong nominal title")
    try expect(ThermalPresentation(state: .fair).title == "偏热", "wrong fair title")
    try expect(ThermalPresentation(state: .serious).title == "严重", "wrong serious title")
    try expect(ThermalPresentation(state: .critical).title == "危急", "wrong critical title")
}

test("usage tracker sorts and keeps seven applications") {
    var tracker = HighUsageTracker()
    let result = tracker.evaluate([
        AppUsage(pid: 1, name: "A", cpuPercent: 20, canTerminate: true),
        AppUsage(pid: 2, name: "B", cpuPercent: 90, canTerminate: true),
        AppUsage(pid: 3, name: "C", cpuPercent: 40, canTerminate: true),
        AppUsage(pid: 4, name: "D", cpuPercent: 60, canTerminate: true),
        AppUsage(pid: 5, name: "E", cpuPercent: 10, canTerminate: true),
        AppUsage(pid: 6, name: "F", cpuPercent: 30, canTerminate: true),
        AppUsage(pid: 7, name: "G", cpuPercent: 50, canTerminate: true),
        AppUsage(pid: 8, name: "H", cpuPercent: 70, canTerminate: true)
    ])
    try expect(result.map(\.name) == ["B", "H", "D", "G", "C", "F", "A"], "expected descending top seven")
}

test("usage tracker requires two samples above eighty percent") {
    var tracker = HighUsageTracker()
    let first = tracker.evaluate([AppUsage(pid: 8, name: "Render", cpuPercent: 81, canTerminate: true)])
    let second = tracker.evaluate([AppUsage(pid: 8, name: "Render", cpuPercent: 82, canTerminate: true)])
    try expect(first[0].isSustainedHigh == false, "first spike must not alert")
    try expect(second[0].isSustainedHigh == true, "second high sample must alert")
}

test("CPU calculator uses process time delta and clamps negative values") {
    try expect(CPUUsageCalculator.percent(previousCPUTime: 2, currentCPUTime: 3, elapsed: 2) == 50, "wrong CPU percent")
    try expect(CPUUsageCalculator.percent(previousCPUTime: 3, currentCPUTime: 2, elapsed: 1) == 0, "negative CPU must clamp")
    try expect(CPUUsageCalculator.percent(previousCPUTime: 1, currentCPUTime: 2, elapsed: 0) == 0, "zero elapsed must return zero")
}

test("thermal notification gate waits ten minutes") {
    var gate = ThermalNotificationGate(cooldown: 600)
    let start = Date(timeIntervalSince1970: 1_000)
    try expect(gate.shouldNotify(state: .serious, now: start), "first serious state must notify")
    try expect(!gate.shouldNotify(state: .critical, now: start.addingTimeInterval(599)), "cooldown must suppress")
    try expect(gate.shouldNotify(state: .serious, now: start.addingTimeInterval(600)), "notification must resume after cooldown")
    try expect(!gate.shouldNotify(state: .fair, now: start.addingTimeInterval(1_200)), "fair state must not notify")
}

test("notifications require an app bundle") {
    try expect(AppBundleEnvironment.supportsNotifications(bundleURL: URL(fileURLWithPath: "/Applications/Owl.app")), "app bundle must support notifications")
    try expect(!AppBundleEnvironment.supportsNotifications(bundleURL: URL(fileURLWithPath: "/tmp/debug/")), "debug executable must skip notifications")
}

test("application usage includes only regular user apps") {
    try expect(ApplicationVisibilityPolicy.shouldInclude(isRegular: true, hasBundleIdentifier: true, isCurrentProcess: false), "regular app must be included")
    try expect(!ApplicationVisibilityPolicy.shouldInclude(isRegular: false, hasBundleIdentifier: true, isCurrentProcess: false), "background service must be excluded")
    try expect(!ApplicationVisibilityPolicy.shouldInclude(isRegular: true, hasBundleIdentifier: false, isCurrentProcess: false), "unidentified process must be excluded")
    try expect(!ApplicationVisibilityPolicy.shouldInclude(isRegular: true, hasBundleIdentifier: true, isCurrentProcess: true), "own process must be excluded")
}

test("application CPU total includes helpers inside its app bundle") {
    let records = [
        ProcessCPURecord(path: "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT", cpuTime: 2),
        ProcessCPURecord(path: "/Applications/ChatGPT.app/Contents/Frameworks/Renderer", cpuTime: 3),
        ProcessCPURecord(path: "/Applications/Figma.app/Contents/MacOS/Figma", cpuTime: 9)
    ]
    try expect(ApplicationCPUAggregator.totalCPUTime(bundlePath: "/Applications/ChatGPT.app", records: records) == 5, "helpers must be included")
}

test("CPU display keeps one decimal below ten percent") {
    try expect(CPUUsageFormatter.string(0.24) == "0.2% CPU", "small usage must remain visible")
    try expect(CPUUsageFormatter.string(8.86) == "8.9% CPU", "single digit usage needs one decimal")
    try expect(CPUUsageFormatter.string(18.4) == "18% CPU", "large usage should be compact")
}

test("monitoring interval slows only for hidden nominal state") {
    try expect(MonitoringIntervalPolicy.interval(windowVisible: false, thermalState: .nominal) == 20, "hidden nominal interval must be twenty seconds")
    try expect(MonitoringIntervalPolicy.interval(windowVisible: true, thermalState: .nominal) == 10, "visible interval must be ten seconds")
    try expect(MonitoringIntervalPolicy.interval(windowVisible: false, thermalState: .fair) == 10, "elevated thermal interval must be ten seconds")
    try expect(MonitoringIntervalPolicy.interval(windowVisible: false, thermalState: .serious) == 10, "serious interval must be ten seconds")
}

test("helper restores after fifteen seconds without heartbeat") {
    var helper = HelperStateMachine(timeout: 15)
    try expect(helper.enable(now: 100, originalValue: 0) == .setSleepDisabled(1), "enable must set one")
    try expect(helper.tick(now: 114.9) == .none, "must remain enabled before timeout")
    try expect(helper.tick(now: 115) == .setSleepDisabled(0), "must restore at timeout")
}

test("helper heartbeat extends the deadline") {
    var helper = HelperStateMachine(timeout: 15)
    _ = helper.enable(now: 100, originalValue: 0)
    helper.heartbeat(now: 110)
    try expect(helper.tick(now: 120) == .none, "heartbeat must extend deadline")
    try expect(helper.tick(now: 125) == .setSleepDisabled(0), "extended deadline must restore")
}

test("helper disable restores original value") {
    var helper = HelperStateMachine(timeout: 15)
    _ = helper.enable(now: 0, originalValue: 1)
    try expect(helper.disable() == .setSleepDisabled(1), "disable must restore original value")
    try expect(helper.isEnabled == false, "helper must be disabled")
}

test("release metadata identifies version 0.1.0 build 2") {
    guard let root = ProcessInfo.processInfo.environment["NO_SLEEP_OWL_ROOT"] else {
        throw TestError.expectation("NO_SLEEP_OWL_ROOT must point to the repository root")
    }
    let plistURL = URL(fileURLWithPath: root).appendingPathComponent("Resources/Info.plist")
    let plistData = try Data(contentsOf: plistURL)
    guard let metadata = try PropertyListSerialization.propertyList(from: plistData, format: nil) as? [String: Any] else {
        throw TestError.expectation("Info.plist must contain a dictionary")
    }
    try expect(metadata["CFBundleShortVersionString"] as? String == "0.1.0", "release version must be 0.1.0")
    try expect(metadata["CFBundleVersion"] as? String == "2", "release build must be 2")
}

if failures > 0 {
    print("\(failures) TEST(S) FAILED")
    exit(1)
}
print("ALL TESTS PASSED")

private enum TestError: Error {
    case failed
    case expectation(String)
}

private final class FakeSleepAssertionController: SleepAssertionControlling {
    let acquiredID: UInt32
    let acquireError: Error?
    let releaseError: Error?
    private(set) var acquireCount = 0
    private(set) var releasedIDs: [UInt32] = []

    init(acquiredID: UInt32 = 1, acquireError: Error? = nil, releaseError: Error? = nil) {
        self.acquiredID = acquiredID
        self.acquireError = acquireError
        self.releaseError = releaseError
    }

    func acquire() throws -> UInt32 {
        acquireCount += 1
        if let acquireError { throw acquireError }
        return acquiredID
    }

    func release(_ assertionID: UInt32) throws {
        if let releaseError { throw releaseError }
        releasedIDs.append(assertionID)
    }
}
