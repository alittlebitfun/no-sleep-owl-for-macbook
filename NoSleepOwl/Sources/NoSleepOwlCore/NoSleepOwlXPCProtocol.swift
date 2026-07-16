import Foundation

@objc public protocol NoSleepOwlXPCProtocol {
    func enable(reply: @escaping (Bool, String?) -> Void)
    func disable(reply: @escaping (Bool, String?) -> Void)
    func heartbeat()
    func status(reply: @escaping (Bool) -> Void)
}

public enum NoSleepOwlService {
    public static let machName = "com.shiying.NoSleepOwl.helper"
    public static let plistName = "com.shiying.NoSleepOwl.helper.plist"
}
