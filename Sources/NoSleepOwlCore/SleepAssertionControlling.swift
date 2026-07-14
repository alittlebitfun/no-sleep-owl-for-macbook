import Foundation

public protocol SleepAssertionControlling: AnyObject {
    func acquire() throws -> UInt32
    func release(_ assertionID: UInt32) throws
}
