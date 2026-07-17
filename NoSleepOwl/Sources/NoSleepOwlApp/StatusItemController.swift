import AppKit
import NoSleepOwlCore

@MainActor
final class StatusItemController: NSObject {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let thermalMonitor: ThermalAppMonitor
    private let preferences: AppPreferences
    private let openWindow: () -> Void
    private let openSettings: () -> Void
    private let quit: () -> Void
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

    init(store: OwlModeStore, launchController: LaunchAtLoginController, thermalMonitor: ThermalAppMonitor, preferences: AppPreferences, openWindow: @escaping () -> Void, openSettings: @escaping () -> Void, quit: @escaping () -> Void) {
        self.store = store
        self.launchController = launchController
        self.thermalMonitor = thermalMonitor
        self.preferences = preferences
        self.openWindow = openWindow
        self.openSettings = openSettings
        self.quit = quit
        super.init()
        if let button = statusItem.button {
            button.target = self
            button.action = #selector(clicked(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
            button.imageScaling = .scaleProportionallyDown
            button.contentTintColor = .white
        }
        statusItem.isVisible = true
        refresh()
    }

    func refresh() {
        let presentation = BirdPresentation(mode: store.mode, language: preferences.snapshot.language)
        let strings = AppStrings(language: preferences.snapshot.language)
        statusItem.button?.title = ""
        statusItem.button?.image = BirdIconRenderer.image(for: store.mode, language: preferences.snapshot.language)
        statusItem.button?.imagePosition = .imageOnly
        statusItem.button?.toolTip = "\(strings.appName) · \(presentation.statusTitle)"
    }

    @objc private func clicked(_ sender: Any?) {
        // Match the familiar menu-bar utility interaction: either mouse button
        // opens the same menu anchored above the status item.
        showMenu(from: nil, event: NSApp.currentEvent)
    }

    private func handle(_ mouseButton: StatusBarMouseButton) {
        switch StatusBarInteraction.action(for: mouseButton) {
        case .openControlWindow: openWindow()
        case .showContextMenu: showMenu(from: nil, event: NSApp.currentEvent)
        }
    }

    private func showMenu(from sender: NSButton?, event: NSEvent?) {
        let snapshot = preferences.snapshot
        let strings = AppStrings(language: snapshot.language)
        let p = BirdPresentation(mode: store.mode, language: snapshot.language)
        let menu = NSMenu()
        let status = NSMenuItem(title: "\(p.emoji)  \(p.statusTitle)", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        let displayMode = MonitoringDisplayPolicy.mode(thermal: snapshot.showsThermalStatus, applications: snapshot.showsHighUsageApps)
        if displayMode != .hidden, let monitorSnapshot = thermalMonitor.latestSnapshot {
            let thermal = ThermalPresentation(state: monitorSnapshot.thermalState, language: snapshot.language)
            let thermalText = snapshot.showsThermalStatus ? thermal.title : strings.applicationUsageTitle
            let top = snapshot.showsHighUsageApps ? (monitorSnapshot.applications.first.map { " · \($0.usage.name) \(CPUUsageFormatter.string($0.usage.cpuPercent))" } ?? "") : ""
            let item = NSMenuItem(title: strings.computerStatus(thermalText, top: top), action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        } else if displayMode != .hidden {
            let item = NSMenuItem(title: strings.computerStatusChecking, action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        menu.addItem(.separator())
        menu.addItem(item(p.toggleTitle, #selector(toggleMode)))
        menu.addItem(item(strings.openApplication, #selector(openControlWindow)))
        menu.addItem(item(strings.settingsMenuTitle, #selector(openSettingsWindow)))
        menu.addItem(.separator())
        let login = item(strings.loginAtStartup, #selector(toggleLogin))
        login.state = launchController.isEnabled ? .on : .off
        menu.addItem(login)
        menu.addItem(.separator())
        menu.addItem(item(strings.quitApplication, #selector(quitApp)))
        if let sender, let event {
            NSMenu.popUpContextMenu(menu, with: event, for: sender)
        } else {
            guard let button = statusItem.button else { return }
            // Pop the menu from the real status-item button so AppKit anchors it
            // above the menu bar icon, like other menu-bar utilities.
            menu.popUp(positioning: nil, at: NSPoint(x: button.bounds.midX, y: button.bounds.minY), in: button)
        }
    }

    private func item(_ title: String, _ action: Selector) -> NSMenuItem {
        let value = NSMenuItem(title: title, action: action, keyEquivalent: "")
        value.target = self
        return value
    }

    @objc private func toggleMode() { store.toggle() }
    @objc private func openControlWindow() { openWindow() }
    @objc private func openSettingsWindow() { openSettings() }
    @objc private func toggleLogin() { try? launchController.setEnabled(!launchController.isEnabled) }
    @objc private func quitApp() { quit() }
}
