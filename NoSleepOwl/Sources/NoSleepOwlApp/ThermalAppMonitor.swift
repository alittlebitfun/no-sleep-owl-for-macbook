import AppKit
import Foundation
import NoSleepOwlCore
import UserNotifications

struct ThermalAppSnapshot {
    let thermalState: OwlThermalState
    let applications: [MonitoredApplication]
    let sampledAt: Date
}

@MainActor
final class ThermalAppMonitor {
    private let sampler = ApplicationUsageSampler()
    private var highUsageTracker = HighUsageTracker()
    private var notificationGate = ThermalNotificationGate()
    private var timer: Timer?
    private var isWindowVisible = false

    private(set) var latestSnapshot: ThermalAppSnapshot?
    var onChange: (() -> Void)?

    func start() {
        if AppBundleEnvironment.supportsNotifications(bundleURL: Bundle.main.bundleURL) {
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
        }
        evaluate()
    }

    func evaluate(now: Date = Date()) {
        let state = currentThermalState()
        let sampled = sampler.sample(now: now)
        let ranked = highUsageTracker.evaluate(sampled.map(\.usage))
        let byPID = Dictionary(uniqueKeysWithValues: sampled.map { ($0.usage.pid, $0) })
        let applications = ranked.compactMap { usage -> MonitoredApplication? in
            guard let source = byPID[usage.pid] else { return nil }
            return MonitoredApplication(usage: usage, icon: source.icon)
        }
        latestSnapshot = ThermalAppSnapshot(thermalState: state, applications: applications, sampledAt: now)
        if AppBundleEnvironment.supportsNotifications(bundleURL: Bundle.main.bundleURL),
           notificationGate.shouldNotify(state: state, now: now) {
            sendThermalNotification(state: state, applications: applications)
        }
        onChange?()
        scheduleNext(for: state)
    }

    func setWindowVisible(_ visible: Bool) {
        isWindowVisible = visible
        guard let state = latestSnapshot?.thermalState else { return }
        scheduleNext(for: state)
    }

    func terminate(pid: pid_t) {
        guard let app = NSRunningApplication(processIdentifier: pid) else { return }
        _ = app.terminate()
    }

    private func currentThermalState() -> OwlThermalState {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: .nominal
        case .fair: .fair
        case .serious: .serious
        case .critical: .critical
        @unknown default: .fair
        }
    }

    private func scheduleNext(for state: OwlThermalState) {
        timer?.invalidate()
        let interval = MonitoringIntervalPolicy.interval(windowVisible: isWindowVisible, thermalState: state)
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated { self?.evaluate() }
        }
    }

    private func sendThermalNotification(state: OwlThermalState, applications: [MonitoredApplication]) {
        let content = UNMutableNotificationContent()
        content.title = state == .critical ? "Mac 热状态危急" : "Mac 温度压力较高"
        if let top = applications.first {
            content.body = "\(top.usage.name) 当前占用约 \(CPUUsageFormatter.string(top.usage.cpuPercent))，请检查通风和运行任务。"
        } else {
            content.body = "请改善通风并检查正在运行的应用。"
        }
        content.sound = .default
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: "NoSleepOwl.thermal", content: content, trigger: nil))
    }
}
