import Foundation
import NoSleepOwlCore
import ServiceManagement

enum PrivilegedSleepError: LocalizedError {
    case needsApproval
    case unavailable(String)
    case requestFailed(String)

    var errorDescription: String? {
        switch self {
        case .needsApproval: return "请先在系统设置中批准合盖守夜辅助程序。"
        case .unavailable(let message): return "辅助程序不可用：\(message)"
        case .requestFailed(let message): return "辅助程序操作失败：\(message)"
        }
    }
}

@MainActor
final class PrivilegedSleepController: SleepAssertionControlling {
    private let idleController = IOKitSleepAssertionController()
    private let service = SMAppService.daemon(plistName: NoSleepOwlService.plistName)
    private var connection: NSXPCConnection?
    private var heartbeatTimer: Timer?

    var statusText: String {
        switch service.status {
        case .enabled: return "辅助程序已批准 · 日常切换无需密码"
        case .requiresApproval: return "等待在系统设置中批准"
        case .notRegistered: return "辅助程序尚未注册"
        case .notFound: return "安装包中未找到辅助程序"
        @unknown default: return "辅助程序状态未知"
        }
    }

    var isReady: Bool { service.status == .enabled }

    func register() throws {
        if service.status == .notRegistered { try service.register() }
        if service.status == .requiresApproval {
            SMAppService.openSystemSettingsLoginItems()
            throw PrivilegedSleepError.needsApproval
        }
        guard service.status == .enabled else { throw PrivilegedSleepError.unavailable(statusText) }
    }

    func acquire() throws -> UInt32 {
        try register()
        let assertionID = try idleController.acquire()
        do {
            try request { proxy, reply in proxy.enable(reply: reply) }
            startHeartbeat()
            return assertionID
        } catch {
            try? idleController.release(assertionID)
            throw error
        }
    }

    func release(_ assertionID: UInt32) throws {
        stopHeartbeat()
        try request { proxy, reply in proxy.disable(reply: reply) }
        try idleController.release(assertionID)
    }

    func refreshStatus() {}

    private func proxy() throws -> NoSleepOwlXPCProtocol {
        if connection == nil {
            let value = NSXPCConnection(machServiceName: NoSleepOwlService.machName, options: .privileged)
            value.remoteObjectInterface = NSXPCInterface(with: NoSleepOwlXPCProtocol.self)
            value.invalidationHandler = { }
            value.resume()
            connection = value
        }
        guard let proxy = connection?.remoteObjectProxyWithErrorHandler({ _ in }) as? NoSleepOwlXPCProtocol else {
            throw PrivilegedSleepError.unavailable("无法建立 XPC 连接")
        }
        return proxy
    }

    private func request(_ operation: (NoSleepOwlXPCProtocol, @escaping (Bool, String?) -> Void) -> Void) throws {
        let remote = try proxy()
        let semaphore = DispatchSemaphore(value: 0)
        var succeeded = false
        var message: String?
        operation(remote) { ok, error in succeeded = ok; message = error; semaphore.signal() }
        guard semaphore.wait(timeout: .now() + 5) == .success else {
            throw PrivilegedSleepError.requestFailed("请求超时")
        }
        guard succeeded else { throw PrivilegedSleepError.requestFailed(message ?? "未知错误") }
    }

    private func startHeartbeat() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let remote = try? self?.proxy() else { return }
                remote.heartbeat()
            }
        }
    }

    private func stopHeartbeat() { heartbeatTimer?.invalidate(); heartbeatTimer = nil }
}
