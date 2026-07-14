import Foundation

public enum HelperAction: Equatable, Sendable {
    case none
    case setSleepDisabled(Int)
}

public struct HelperStateMachine: Sendable {
    public private(set) var isEnabled = false
    private let timeout: TimeInterval
    private var lastHeartbeat: TimeInterval?
    private var originalValue = 0

    public init(timeout: TimeInterval = 15) { self.timeout = timeout }

    public mutating func enable(now: TimeInterval, originalValue: Int) -> HelperAction {
        self.originalValue = originalValue
        lastHeartbeat = now
        isEnabled = true
        return .setSleepDisabled(1)
    }

    public mutating func heartbeat(now: TimeInterval) {
        guard isEnabled else { return }
        lastHeartbeat = now
    }

    public mutating func tick(now: TimeInterval) -> HelperAction {
        guard isEnabled, let lastHeartbeat, now - lastHeartbeat >= timeout else { return .none }
        return disable()
    }

    public mutating func disable() -> HelperAction {
        guard isEnabled else { return .none }
        isEnabled = false
        lastHeartbeat = nil
        return .setSleepDisabled(originalValue)
    }
}
