import Foundation

public struct ThermalPresentation: Sendable {
    public let title: String
    public let detail: String

    public init(state: OwlThermalState) {
        switch state {
        case .nominal:
            title = "正常"; detail = "电脑运行状态良好"
        case .fair:
            title = "偏热"; detail = "建议留意高占用应用"
        case .serious:
            title = "严重"; detail = "请改善通风并检查高占用应用"
        case .critical:
            title = "危急"; detail = "正在执行过热保护"
        }
    }
}

public struct AppUsage: Sendable, Equatable {
    public let pid: Int32
    public let name: String
    public let cpuPercent: Double
    public let canTerminate: Bool
    public let isSustainedHigh: Bool

    public init(pid: Int32, name: String, cpuPercent: Double, canTerminate: Bool, isSustainedHigh: Bool = false) {
        self.pid = pid
        self.name = name
        self.cpuPercent = cpuPercent
        self.canTerminate = canTerminate
        self.isSustainedHigh = isSustainedHigh
    }
}

public struct HighUsageTracker: Sendable {
    private var consecutiveHighSamples: [Int32: Int] = [:]

    public init() {}

    public mutating func evaluate(_ usages: [AppUsage]) -> [AppUsage] {
        let active = Set(usages.map(\.pid))
        consecutiveHighSamples = consecutiveHighSamples.filter { active.contains($0.key) }
        let marked = usages.map { usage in
            let count = usage.cpuPercent > 80 ? (consecutiveHighSamples[usage.pid] ?? 0) + 1 : 0
            consecutiveHighSamples[usage.pid] = count
            return AppUsage(
                pid: usage.pid,
                name: usage.name,
                cpuPercent: usage.cpuPercent,
                canTerminate: usage.canTerminate,
                isSustainedHigh: count >= 2
            )
        }
        return Array(marked.sorted { $0.cpuPercent > $1.cpuPercent }.prefix(3))
    }
}

public enum CPUUsageCalculator {
    public static func percent(previousCPUTime: Double, currentCPUTime: Double, elapsed: Double) -> Double {
        guard elapsed > 0 else { return 0 }
        return max(0, (currentCPUTime - previousCPUTime) / elapsed * 100)
    }
}

public struct ThermalNotificationGate: Sendable {
    private var lastNotification: Date?
    private let cooldown: TimeInterval

    public init(cooldown: TimeInterval = 600) { self.cooldown = cooldown }

    public mutating func shouldNotify(state: OwlThermalState, now: Date) -> Bool {
        guard state == .serious || state == .critical else { return false }
        guard lastNotification == nil || now.timeIntervalSince(lastNotification!) >= cooldown else { return false }
        lastNotification = now
        return true
    }
}

public enum AppBundleEnvironment {
    public static func supportsNotifications(bundleURL: URL) -> Bool {
        bundleURL.pathExtension == "app"
    }
}

public enum ApplicationVisibilityPolicy {
    public static func shouldInclude(isRegular: Bool, hasBundleIdentifier: Bool, isCurrentProcess: Bool) -> Bool {
        isRegular && hasBundleIdentifier && !isCurrentProcess
    }
}
