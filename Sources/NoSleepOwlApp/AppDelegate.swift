import AppKit
import NoSleepOwlCore

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let store = OwlModeStore(controller: IOKitSleepAssertionController())
    private var statusController: StatusItemController!
    private var windowController: ControlWindowController!
    private let launchController = LaunchAtLoginController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        windowController = ControlWindowController(store: store, launchController: launchController)
        statusController = StatusItemController(
            store: store,
            launchController: launchController,
            openWindow: { [weak self] in self?.windowController.show() },
            quit: { NSApplication.shared.terminate(nil) }
        )
        store.onChange = { [weak self] in
            self?.statusController.refresh()
            self?.windowController.refresh()
        }
        if ProcessInfo.processInfo.arguments.contains("--open-window") {
            windowController.show()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }
}
