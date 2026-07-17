import AppKit
import NoSleepOwlCore

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let sleepController = PrivilegedSleepController()
    private let preferences = AppPreferences()
    private lazy var store = OwlModeStore(controller: sleepController)
    private lazy var safetyMonitor = SafetyMonitor(store: store, preferences: preferences)
    private lazy var thermalMonitor = ThermalAppMonitor()
    private var displayController: DisplayLocationController!
    private var windowController: ControlWindowController!
    private var settingsController: SettingsWindowController!
    private let launchController = LaunchAtLoginController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        settingsController = SettingsWindowController(preferences: preferences)
        windowController = ControlWindowController(store: store, launchController: launchController, sleepController: sleepController, safetyMonitor: safetyMonitor, thermalMonitor: thermalMonitor, preferences: preferences)
        displayController = DisplayLocationController(store: store, launchController: launchController, thermalMonitor: thermalMonitor, preferences: preferences, openWindow: { [weak self] in self?.windowController.show() }, openSettings: { [weak self] in self?.settingsController.show() }, quit: { NSApplication.shared.terminate(nil) })
        windowController.onOpenSettings = { [weak self] in self?.settingsController.show() }
        windowController.onVisibilityChange = { [weak self] visible in self?.displayController.setMainWindowVisible(visible) }
        configureMainMenu()
        store.onChange = { [weak self] in
            self?.displayController.refreshStatusItem()
            self?.windowController.refresh()
        }
        thermalMonitor.onChange = { [weak self] in
            self?.windowController.refresh()
            self?.displayController.refreshStatusItem()
        }
        preferences.onChange = { [weak self] in
            guard let self else { return }
            thermalMonitor.configure(preferences.snapshot)
            store.setLanguage(preferences.snapshot.language)
            windowController.refresh()
            settingsController.refresh()
            displayController.apply(preferences.snapshot)
        }
        store.setLanguage(preferences.snapshot.language)
        thermalMonitor.configure(preferences.snapshot)
        displayController.apply(preferences.snapshot)
        if ProcessInfo.processInfo.arguments.contains("--open-window") {
            windowController.show()
        }
        if ProcessInfo.processInfo.arguments.contains("--open-settings") {
            settingsController.show()
        }
        safetyMonitor.start()
        thermalMonitor.start()
    }

    private func configureMainMenu() {
        let menu = NSMenu()
        let appMenu = NSMenu()
        let appItem = NSMenuItem(title: AppStrings(language: preferences.snapshot.language).appName, action: nil, keyEquivalent: "")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        appMenu.addItem(withTitle: "设置…", action: #selector(openSettingsFromMenu), keyEquivalent: ",")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出", action: #selector(quitFromMenu), keyEquivalent: "q")
        appMenu.items.forEach { $0.target = self }
        let windowMenu = NSMenu(title: "窗口")
        let windowItem = NSMenuItem(title: "窗口", action: nil, keyEquivalent: "")
        windowItem.submenu = windowMenu
        menu.addItem(windowItem)
        NSApp.mainMenu = menu
    }

    @objc private func openSettingsFromMenu() { settingsController.show() }
    @objc private func quitFromMenu() { NSApp.terminate(nil) }

    func applicationWillTerminate(_ notification: Notification) {
        store.shutdown()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        windowController.show()
        return ApplicationReopenPolicy.opensControlWindow
    }
}
