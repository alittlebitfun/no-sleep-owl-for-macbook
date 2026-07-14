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

test("IOKit assertion prevents only idle system sleep") {
    try expect(IOKitSleepAssertionController.assertionType == "PreventUserIdleSystemSleep", "wrong assertion type")
}

test("bird and owl presentations are distinct") {
    try expect(BirdPresentation(mode: .bird).statusTitle == "小鸟可以休息", "wrong bird title")
    try expect(BirdPresentation(mode: .owl).statusTitle == "猫头鹰正在守夜", "wrong owl title")
    try expect(BirdPresentation(mode: .bird).toggleTitle == "切换到猫头鹰模式", "wrong bird action")
    try expect(BirdPresentation(mode: .owl).toggleTitle == "切换到小鸟模式", "wrong owl action")
}

test("duration formatter covers seconds minutes and hours") {
    try expect(OwlDurationFormatter.string(seconds: 0) == "00:00", "wrong zero duration")
    try expect(OwlDurationFormatter.string(seconds: 65) == "01:05", "wrong minute duration")
    try expect(OwlDurationFormatter.string(seconds: 3661) == "01:01:01", "wrong hour duration")
}

test("closed-lid helper restores the original sleep setting") {
    let script = ClosedLidHelperScript.make(markerPath: "/tmp/owl marker", appPID: 123, restoreValue: 0)
    try expect(script.contains("pmset -a disablesleep 1"), "helper must disable lid sleep")
    try expect(script.contains("kill -0 123"), "helper must monitor the app")
    try expect(script.contains("pmset -a disablesleep 0"), "helper must restore the original value")
    try expect(script.contains("'/tmp/owl marker'"), "marker path must be shell quoted")
    try expect(script.contains("trap cleanup"), "helper must clean up on termination")
    try expect(script.contains("/bin/test -e"), "helper must use the macOS test binary path")
    try expect(!script.contains("/usr/bin/test"), "helper must not use a missing binary")
}

test("closed-lid launcher avoids nohup in administrator context") {
    let command = ClosedLidHelperScript.launchCommand(script: "echo ready", logPath: "/tmp/owl.log")
    try expect(!command.contains("nohup"), "macOS nohup cannot detach in the authorization shell")
    try expect(command.hasSuffix("</dev/null &"), "helper must run in background with detached input")
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
