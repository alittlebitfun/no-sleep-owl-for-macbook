import AppKit
import NoSleepOwlCore

@MainActor
final class DisplayLocationController {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let thermalMonitor: ThermalAppMonitor
    private let preferences: AppPreferences
    private let openWindow: () -> Void
    private let openSettings: () -> Void
    private let quit: () -> Void
    private var statusController: StatusItemController?

    init(store: OwlModeStore, launchController: LaunchAtLoginController, thermalMonitor: ThermalAppMonitor, preferences: AppPreferences, openWindow: @escaping () -> Void, openSettings: @escaping () -> Void, quit: @escaping () -> Void) {
        self.store = store
        self.launchController = launchController
        self.thermalMonitor = thermalMonitor
        self.preferences = preferences
        self.openWindow = openWindow
        self.openSettings = openSettings
        self.quit = quit
    }

    func apply(_ snapshot: AppPreferenceSnapshot) {
        NSApp.setActivationPolicy(snapshot.showsDockIcon ? .regular : .accessory)
        if snapshot.showsStatusBarIcon {
            if statusController == nil {
                statusController = StatusItemController(store: store, launchController: launchController, thermalMonitor: thermalMonitor, preferences: preferences, openWindow: openWindow, openSettings: openSettings, quit: quit)
            } else {
                statusController?.refresh()
            }
        } else {
            statusController = nil
        }
    }

    func refreshStatusItem() { statusController?.refresh() }

    func setMainWindowVisible(_ visible: Bool) {
        if visible {
            NSApp.setActivationPolicy(.regular)
        } else {
            apply(preferences.snapshot)
        }
    }
}
