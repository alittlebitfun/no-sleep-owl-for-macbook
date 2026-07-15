import AppKit
import Darwin
import NoSleepOwlCore

struct MonitoredApplication {
    let usage: AppUsage
    let icon: NSImage?
}

@MainActor
final class ApplicationUsageSampler {
    private struct PreviousSample {
        let cpuTime: Double
        let date: Date
    }

    private var previous: [pid_t: PreviousSample] = [:]
    private let protectedBundleIDs: Set<String> = [
        "com.apple.finder",
        "com.apple.systempreferences",
        "com.apple.SystemSettings",
        "com.apple.loginwindow"
    ]

    func sample(now: Date = Date()) -> [MonitoredApplication] {
        let ownPID = ProcessInfo.processInfo.processIdentifier
        let processRecords = allProcessRecords()
        var activePIDs = Set<pid_t>()
        let results = NSWorkspace.shared.runningApplications.compactMap { app -> MonitoredApplication? in
            let pid = app.processIdentifier
            guard ApplicationVisibilityPolicy.shouldInclude(
                    isRegular: app.activationPolicy == .regular,
                    hasBundleIdentifier: app.bundleIdentifier != nil,
                    isCurrentProcess: pid == ownPID
                  ),
                  let name = app.localizedName,
                  !name.isEmpty,
                  let bundlePath = app.bundleURL?.path else { return nil }
            let cpuTime = ApplicationCPUAggregator.totalCPUTime(bundlePath: bundlePath, records: processRecords)
            activePIDs.insert(pid)
            let cpu: Double
            if let old = previous[pid] {
                cpu = CPUUsageCalculator.percent(
                    previousCPUTime: old.cpuTime,
                    currentCPUTime: cpuTime,
                    elapsed: now.timeIntervalSince(old.date)
                )
            } else {
                cpu = 0
            }
            previous[pid] = PreviousSample(cpuTime: cpuTime, date: now)
            let canTerminate = app.bundleIdentifier.map { !protectedBundleIDs.contains($0) } ?? false
            return MonitoredApplication(
                usage: AppUsage(pid: pid, name: name, cpuPercent: cpu, canTerminate: canTerminate),
                icon: app.icon
            )
        }
        previous = previous.filter { activePIDs.contains($0.key) }
        return results
    }

    private func allProcessRecords() -> [ProcessCPURecord] {
        var pids = [pid_t](repeating: 0, count: 8_192)
        let byteCount = Int32(pids.count * MemoryLayout<pid_t>.size)
        let count = proc_listallpids(&pids, byteCount)
        guard count > 0 else { return [] }
        return pids.prefix(Int(count)).compactMap { pid in
            guard pid > 0, let path = executablePath(pid: pid), let time = cpuTime(pid: pid) else { return nil }
            return ProcessCPURecord(path: path, cpuTime: time)
        }
    }

    private func executablePath(pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: 4_096)
        let length = buffer.withUnsafeMutableBytes { bytes in
            proc_pidpath(pid, bytes.baseAddress, UInt32(bytes.count))
        }
        guard length > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    private func cpuTime(pid: pid_t) -> Double? {
        var info = proc_taskinfo()
        let size = Int32(MemoryLayout<proc_taskinfo>.size)
        let read = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &info, size)
        guard read == size else { return nil }
        return Double(info.pti_total_user + info.pti_total_system) / 1_000_000_000
    }
}
