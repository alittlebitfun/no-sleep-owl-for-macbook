import Foundation
import IOKit.pwr_mgt

public struct SleepAssertionError: LocalizedError {
    public let code: IOReturn
    public var errorDescription: String? { "电源管理操作失败（\(code)）" }
}

public final class IOKitSleepAssertionController: SleepAssertionControlling {
    public static let assertionType = "PreventUserIdleSystemSleep"

    public init() {}

    public func acquire() throws -> UInt32 {
        var assertionID: IOPMAssertionID = 0
        let result = IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "不休眠猫头鹰正在守夜" as CFString,
            &assertionID
        )
        guard result == kIOReturnSuccess else { throw SleepAssertionError(code: result) }
        return assertionID
    }

    public func release(_ assertionID: UInt32) throws {
        let result = IOPMAssertionRelease(assertionID)
        guard result == kIOReturnSuccess else { throw SleepAssertionError(code: result) }
    }
}
