import Foundation
import NoSleepOwlCore
import Darwin

final class HelperService: NSObject, NoSleepOwlXPCProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var state = HelperStateMachine(timeout: 15)
    private var timer: DispatchSourceTimer?

    func startTimer() {
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in self?.tick() }
        self.timer = timer
        timer.resume()
    }

    func enable(reply: @escaping (Bool, String?) -> Void) {
        lock.lock()
        let action = state.enable(now: Date().timeIntervalSince1970, originalValue: currentSleepDisabled())
        lock.unlock()
        reply(apply(action), nil)
    }

    func disable(reply: @escaping (Bool, String?) -> Void) {
        reply(disableAndRestore(), nil)
    }

    func heartbeat() {
        lock.lock()
        state.heartbeat(now: Date().timeIntervalSince1970)
        lock.unlock()
    }

    func status(reply: @escaping (Bool) -> Void) {
        lock.lock(); let enabled = state.isEnabled; lock.unlock()
        reply(enabled)
    }

    func clientDisconnected() { _ = disableAndRestore() }

    private func tick() {
        lock.lock(); let action = state.tick(now: Date().timeIntervalSince1970); lock.unlock()
        _ = apply(action)
    }

    private func disableAndRestore() -> Bool {
        lock.lock(); let action = state.disable(); lock.unlock()
        return apply(action)
    }

    private func apply(_ action: HelperAction) -> Bool {
        switch action {
        case .none: return true
        case .setSleepDisabled(let value): return runPMSet(value)
        }
    }

    private func runPMSet(_ value: Int) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-a", "disablesleep", String(value)]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do { try process.run(); process.waitUntilExit(); return process.terminationStatus == 0 }
        catch { return false }
    }

    private func currentSleepDisabled() -> Int {
        let process = Process(); let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-g"]; process.standardOutput = pipe
        try? process.run(); process.waitUntilExit()
        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return output.contains("SleepDisabled\t\t1") ? 1 : 0
    }
}

final class HelperListenerDelegate: NSObject, NSXPCListenerDelegate, @unchecked Sendable {
    private let service: HelperService
    init(service: HelperService) { self.service = service }

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        guard connection.effectiveUserIdentifier != 0, trustedClientPath(connection.processIdentifier) else { return false }
        connection.exportedInterface = NSXPCInterface(with: NoSleepOwlXPCProtocol.self)
        connection.exportedObject = service
        connection.invalidationHandler = { [weak service] in service?.clientDisconnected() }
        connection.interruptionHandler = { [weak service] in service?.clientDisconnected() }
        connection.resume()
        return true
    }

    private func trustedClientPath(_ pid: pid_t) -> Bool {
        var buffer = [CChar](repeating: 0, count: 4096)
        let length = buffer.withUnsafeMutableBytes { bytes in
            proc_pidpath(pid, bytes.baseAddress, UInt32(bytes.count))
        }
        guard length > 0 else { return false }
        let path = String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return path == "/Applications/不休眠猫头鹰.app/Contents/MacOS/NoSleepOwlApp"
    }
}
