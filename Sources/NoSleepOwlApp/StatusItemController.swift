import AppKit
import NoSleepOwlCore

@MainActor
final class StatusItemController: NSObject {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let openWindow: () -> Void
    private let quit: () -> Void
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

    init(store: OwlModeStore, launchController: LaunchAtLoginController, openWindow: @escaping () -> Void, quit: @escaping () -> Void) {
        self.store = store
        self.launchController = launchController
        self.openWindow = openWindow
        self.quit = quit
        super.init()
        if let button = statusItem.button {
            button.target = self
            button.action = #selector(clicked(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        refresh()
    }

    func refresh() {
        let presentation = BirdPresentation(mode: store.mode)
        statusItem.button?.title = presentation.emoji
        statusItem.button?.toolTip = "不休眠猫头鹰 · \(presentation.statusTitle)"
    }

    @objc private func clicked(_ sender: Any?) {
        if NSApp.currentEvent?.type == .rightMouseUp { showMenu() }
        else { store.toggle() }
    }

    private func showMenu() {
        let p = BirdPresentation(mode: store.mode)
        let menu = NSMenu()
        let status = NSMenuItem(title: "\(p.emoji)  \(p.statusTitle)", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        menu.addItem(.separator())
        menu.addItem(item(p.toggleTitle, #selector(toggleMode)))
        menu.addItem(item("打开不休眠猫头鹰…", #selector(openControlWindow)))
        menu.addItem(.separator())
        let login = item("登录时自动启动", #selector(toggleLogin))
        login.state = launchController.isEnabled ? .on : .off
        menu.addItem(login)
        menu.addItem(.separator())
        menu.addItem(item("退出不休眠猫头鹰", #selector(quitApp)))
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    private func item(_ title: String, _ action: Selector) -> NSMenuItem {
        let value = NSMenuItem(title: title, action: action, keyEquivalent: "")
        value.target = self
        return value
    }

    @objc private func toggleMode() { store.toggle() }
    @objc private func openControlWindow() { openWindow() }
    @objc private func toggleLogin() { try? launchController.setEnabled(!launchController.isEnabled) }
    @objc private func quitApp() { quit() }
}
