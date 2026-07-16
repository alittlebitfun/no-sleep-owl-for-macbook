import Foundation

public enum BirdMode: Equatable, Sendable {
    case bird
    case owl
}

@MainActor
public final class OwlModeStore {
    public private(set) var mode: BirdMode = .bird
    public private(set) var startedAt: Date?
    public private(set) var errorMessage: String?
    public var onChange: (() -> Void)?

    private let controller: SleepAssertionControlling
    private let now: () -> Date
    private var assertionID: UInt32?
    private var language: AppLanguage = .zhHans

    public init(
        controller: SleepAssertionControlling,
        now: @escaping () -> Date = Date.init
    ) {
        self.controller = controller
        self.now = now
    }

    public func toggle() {
        errorMessage = nil
        switch mode {
        case .bird:
            do {
                assertionID = try controller.acquire()
                startedAt = now()
                mode = .owl
            } catch {
                errorMessage = language == .en ? "Could not enable owl mode. Please try again." : error.localizedDescription
            }
        case .owl:
            guard let assertionID else { return }
            do {
                try controller.release(assertionID)
                self.assertionID = nil
                startedAt = nil
                mode = .bird
            } catch {
                errorMessage = language == .en ? "Could not restore sleep. Please try again." : "未能恢复休息，请再试一次。"
            }
        }
        onChange?()
    }

    public func setLanguage(_ value: AppLanguage) { language = value; onChange?() }

    public func setMessage(_ message: String?) {
        errorMessage = message
        onChange?()
    }

    public func shutdown() {
        guard let assertionID else { return }
        do {
            try controller.release(assertionID)
            self.assertionID = nil
            startedAt = nil
            mode = .bird
            errorMessage = nil
        } catch {
            errorMessage = language == .en ? "Could not release sleep control before quitting." : "退出前未能释放休眠控制。"
        }
        onChange?()
    }
}
