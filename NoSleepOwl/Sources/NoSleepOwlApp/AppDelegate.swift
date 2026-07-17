import AppKit
import NoSleepOwlCore

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let sleepController = PrivilegedSleepController()
    private let preferences = AppPreferences()
    private lazy var store = OwlModeStore(controller: sleepController)
    private lazy var safetyMonitor = SafetyMonitor(store: store, preferences: preferences)
    private lazy var thermalMonitor = ThermalAppMonitor()
    private var statusController: StatusItemController!
    private var windowController: ControlWindowController!
    private var settingsController: SettingsWindowController!
    private let launchController = LaunchAtLoginController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        settingsController = SettingsWindowController(preferences: preferences)
        windowController = ControlWindowController(store: store, launchController: launchController, sleepController: sleepController, safetyMonitor: safetyMonitor, thermalMonitor: thermalMonitor, preferences: preferences)
        statusController = StatusItemController(
            store: store,
            launchController: launchController,
            thermalMonitor: thermalMonitor,
            preferences: preferences,
            openWindow: { [weak self] in self?.windowController.show() },
            openSettings: { [weak self] in self?.settingsController.show() },
            quit: { NSApplication.shared.terminate(nil) }
        )
        store.onChange = { [weak self] in
            self?.statusController.refresh()
            self?.windowController.refresh()
        }
        thermalMonitor.onChange = { [weak self] in
            self?.windowController.refresh()
            self?.statusController.refresh()
        }
        preferences.onChange = { [weak self] in
            guard let self else { return }
            thermalMonitor.configure(preferences.snapshot)
            store.setLanguage(preferences.snapshot.language)
            windowController.refresh()
            settingsController.refresh()
            statusController.refresh()
        }
        store.setLanguage(preferences.snapshot.language)
        thermalMonitor.configure(preferences.snapshot)
        if ProcessInfo.processInfo.arguments.contains("--open-window") {
            windowController.show()
        }
        if ProcessInfo.processInfo.arguments.contains("--open-settings") {
            settingsController.show()
        }
        safetyMonitor.start()
        thermalMonitor.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        windowController.show()
        return ApplicationReopenPolicy.opensControlWindow
    }
}
