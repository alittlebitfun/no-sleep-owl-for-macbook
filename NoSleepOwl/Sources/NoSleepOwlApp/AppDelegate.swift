import AppKit
import NoSleepOwlCore

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let sleepController = PrivilegedSleepController()
    private lazy var store = OwlModeStore(controller: sleepController)
    private lazy var safetyMonitor = SafetyMonitor(store: store)
    private var statusController: StatusItemController!
    private var windowController: ControlWindowController!
    private let launchController = LaunchAtLoginController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        windowController = ControlWindowController(store: store, launchController: launchController, sleepController: sleepController, safetyMonitor: safetyMonitor)
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
        safetyMonitor.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }
}
