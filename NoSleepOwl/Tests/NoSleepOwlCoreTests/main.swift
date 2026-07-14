import Foundation
import NoSleepOwlCore

private var failures = 0

@MainActor
private func test(_ name: String, _ body: () throws -> Void) {
    do {
        try body()
        print("PASS \(name)")
    } catch {
        failures += 1
        print("FAIL \(name): \(error)")
    }
}

@MainActor
private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else { throw TestError.expectation(message) }
}

test("starts in bird mode") {
    let store = OwlModeStore(controller: FakeSleepAssertionController())
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.startedAt == nil, "expected no start date")
}

test("toggle acquires before entering owl mode") {
    let controller = FakeSleepAssertionController(acquiredID: 42)
    let now = Date(timeIntervalSince1970: 100)
    let store = OwlModeStore(controller: controller, now: { now })
    store.toggle()
    try expect(store.mode == .owl, "expected owl mode")
    try expect(store.startedAt == now, "expected captured start date")
    try expect(controller.acquireCount == 1, "expected one acquire")
    try expect(store.errorMessage == nil, "expected no error")
}

test("acquire failure keeps bird mode") {
    let store = OwlModeStore(controller: FakeSleepAssertionController(acquireError: TestError.failed))
    store.toggle()
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.errorMessage != nil, "expected error message")
}

test("toggle releases before returning to bird mode") {
    let controller = FakeSleepAssertionController(acquiredID: 7)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.toggle()
    try expect(store.mode == .bird, "expected bird mode")
    try expect(store.startedAt == nil, "expected cleared start date")
    try expect(controller.releasedIDs == [7], "expected assertion 7 released")
}

test("release failure keeps owl mode") {
    let controller = FakeSleepAssertionController(acquiredID: 9, releaseError: TestError.failed)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.toggle()
    try expect(store.mode == .owl, "expected owl mode")
    try expect(store.errorMessage != nil, "expected error message")
}

test("shutdown releases active assertion") {
    let controller = FakeSleepAssertionController(acquiredID: 11)
    let store = OwlModeStore(controller: controller)
    store.toggle()
    store.shutdown()
    try expect(controller.releasedIDs == [11], "expected assertion 11 released")
    try expect(store.mode == .bird, "expected bird mode")
}

if failures > 0 {
    print("\(failures) TEST(S) FAILED")
    exit(1)
}
print("ALL TESTS PASSED")

private enum TestError: Error {
    case failed
    case expectation(String)
}

private final class FakeSleepAssertionController: SleepAssertionControlling {
    let acquiredID: UInt32
    let acquireError: Error?
    let releaseError: Error?
    private(set) var acquireCount = 0
    private(set) var releasedIDs: [UInt32] = []

    init(acquiredID: UInt32 = 1, acquireError: Error? = nil, releaseError: Error? = nil) {
        self.acquiredID = acquiredID
        self.acquireError = acquireError
        self.releaseError = releaseError
    }

    func acquire() throws -> UInt32 {
        acquireCount += 1
        if let acquireError { throw acquireError }
        return acquiredID
    }

    func release(_ assertionID: UInt32) throws {
        if let releaseError { throw releaseError }
        releasedIDs.append(assertionID)
    }
}
