import Foundation
import IOKit.ps
import NoSleepOwlCore

@MainActor
final class SafetyMonitor {
    private let store: OwlModeStore
    private let preferences: AppPreferences
    private var policyEngine = SafetyPolicy()
    private var timer: Timer?

    var powerPolicy: PowerPolicy {
        get { PowerPolicy(rawValue: UserDefaults.standard.string(forKey: "powerPolicy") ?? "acOnly") ?? .acOnly }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "powerPolicy") }
    }

    init(store: OwlModeStore, preferences: AppPreferences) { self.store = store; self.preferences = preferences }

    func start() {
        timer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.evaluate() }
        }
    }

    func evaluate() {
        guard store.mode == .owl else { return }
        let power = currentPower()
        let thermal: OwlThermalState = switch ProcessInfo.processInfo.thermalState {
        case .nominal: .nominal
        case .fair: .fair
        case .serious: .serious
        case .critical: .critical
        @unknown default: .fair
        }
        let strings = AppStrings(language: preferences.snapshot.language)
        switch policyEngine.evaluate(SafetySnapshot(policy: powerPolicy, isOnACPower: power.ac, batteryPercent: power.percent, thermalState: thermal)) {
        case .none: break
        case .warn(.lowBattery): store.setMessage(strings.lowBatteryWarning)
        case .warn(.thermalSerious): store.setMessage(strings.thermalWarning)
        case .exitOwl(let reason):
            store.toggle()
            store.setMessage(message(for: reason, strings: strings))
        }
    }

    private func message(for reason: SafetyExitReason, strings: AppStrings) -> String {
        switch reason {
        case .powerDisconnected: strings.powerDisconnected
        case .criticalBattery: strings.criticalBattery
        case .thermalCritical: strings.thermalCritical
        }
    }

    private func currentPower() -> (ac: Bool, percent: Int) {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef],
              let source = sources.first,
              let description = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any] else {
            return (true, 100)
        }
        let state = description[kIOPSPowerSourceStateKey] as? String
        let current = description[kIOPSCurrentCapacityKey] as? Int ?? 100
        let maximum = description[kIOPSMaxCapacityKey] as? Int ?? 100
        return (state == kIOPSACPowerValue, maximum > 0 ? current * 100 / maximum : 100)
    }
}
