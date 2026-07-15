import Foundation

public enum PowerPolicy: String, Codable, CaseIterable, Sendable {
    case acOnly
    case allowBattery
}

public enum OwlThermalState: Sendable, Equatable {
    case nominal, fair, serious, critical
}

public struct SafetySnapshot: Sendable {
    public let policy: PowerPolicy
    public let isOnACPower: Bool
    public let batteryPercent: Int
    public let thermalState: OwlThermalState

    public init(policy: PowerPolicy, isOnACPower: Bool, batteryPercent: Int, thermalState: OwlThermalState) {
        self.policy = policy
        self.isOnACPower = isOnACPower
        self.batteryPercent = batteryPercent
        self.thermalState = thermalState
    }
}

public enum SafetyWarning: Equatable, Sendable { case lowBattery, thermalSerious }
public enum SafetyExitReason: Equatable, Sendable { case powerDisconnected, criticalBattery, thermalCritical }
public enum SafetyDecision: Equatable, Sendable {
    case none
    case warn(SafetyWarning)
    case exitOwl(reason: SafetyExitReason)
}

public struct SafetyPolicy: Sendable {
    private var warnedBattery = false
    private var warnedThermal = false

    public init() {}

    public mutating func evaluate(_ snapshot: SafetySnapshot) -> SafetyDecision {
        if snapshot.thermalState == .critical { return .exitOwl(reason: .thermalCritical) }
        if !snapshot.isOnACPower && snapshot.batteryPercent <= 10 { return .exitOwl(reason: .criticalBattery) }
        if snapshot.policy == .acOnly && !snapshot.isOnACPower { return .exitOwl(reason: .powerDisconnected) }
        if snapshot.thermalState == .serious && !warnedThermal {
            warnedThermal = true
            return .warn(.thermalSerious)
        }
        if !snapshot.isOnACPower && snapshot.batteryPercent <= 20 && !warnedBattery {
            warnedBattery = true
            return .warn(.lowBattery)
        }
        return .none
    }
}
