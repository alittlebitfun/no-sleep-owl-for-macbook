import Foundation
import NoSleepOwlCore

enum ClosedLidControllerError: LocalizedError {
    case authorizationFailed(String)
    case helperDidNotStart

    var errorDescription: String? {
        switch self {
        case .authorizationFailed: return "需要管理员授权才能开启合盖守夜。"
        case .helperDidNotStart: return "合盖守夜未能启动，请稍后重试。"
        }
    }
}

final class ClosedLidSleepAssertionController: SleepAssertionControlling {
    private let idleController = IOKitSleepAssertionController()
    private var markerPath: String?

    func acquire() throws -> UInt32 {
        let assertionID = try idleController.acquire()
        let originalValue = currentSleepDisabledValue()
        let marker = FileManager.default.temporaryDirectory
            .appendingPathComponent("com.shiying.NoSleepOwl.\(ProcessInfo.processInfo.processIdentifier).\(UUID().uuidString).awake")
            .path
        FileManager.default.createFile(atPath: marker, contents: Data())

        let helper = ClosedLidHelperScript.make(
            markerPath: marker,
            appPID: ProcessInfo.processInfo.processIdentifier,
            restoreValue: originalValue
        )
        let logPath = FileManager.default.temporaryDirectory.appendingPathComponent("NoSleepOwl-helper.log").path
        let command = ClosedLidHelperScript.launchCommand(script: helper, logPath: logPath)
        let appleScript = "do shell script \"\(appleScriptEscape(command))\" with administrator privileges"
        var error: NSDictionary?
        let result = NSAppleScript(source: appleScript)?.executeAndReturnError(&error)
        guard result != nil, error == nil else {
            try? FileManager.default.removeItem(atPath: marker)
            try? idleController.release(assertionID)
            throw ClosedLidControllerError.authorizationFailed(error?.description ?? "cancelled")
        }
        markerPath = marker
        guard waitForSleepDisabled(1, timeout: 5) else {
            try? FileManager.default.removeItem(atPath: marker)
            try? idleController.release(assertionID)
            throw ClosedLidControllerError.helperDidNotStart
        }
        return assertionID
    }

    func release(_ assertionID: UInt32) throws {
        if let markerPath { try? FileManager.default.removeItem(atPath: markerPath) }
        markerPath = nil
        _ = waitForSleepDisabled(0, timeout: 5)
        try idleController.release(assertionID)
    }

    private func currentSleepDisabledValue() -> Int {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-g"]
        process.standardOutput = pipe
        try? process.run()
        process.waitUntilExit()
        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return output.contains("SleepDisabled\t\t1") ? 1 : 0
    }

    private func waitForSleepDisabled(_ value: Int, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if currentSleepDisabledValue() == value { return true }
            Thread.sleep(forTimeInterval: 0.15)
        } while Date() < deadline
        return false
    }

    private func appleScriptEscape(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }
}
