import AppKit
import NoSleepOwlCore

@MainActor
final class StatusItemController: NSObject {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let thermalMonitor: ThermalAppMonitor
    private let openWindow: () -> Void
    private let quit: () -> Void
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

    init(store: OwlModeStore, launchController: LaunchAtLoginController, thermalMonitor: ThermalAppMonitor, openWindow: @escaping () -> Void, quit: @escaping () -> Void) {
        self.store = store
        self.launchController = launchController
        self.thermalMonitor = thermalMonitor
        self.openWindow = openWindow
        self.quit = quit
        super.init()
        if let button = statusItem.button {
            button.target = self
            button.action = #selector(clicked(_:))
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        if StatusItemPlacementPolicy.persistsCustomPosition {
            statusItem.autosaveName = "NoSleepOwl.StatusItem.v2"
        }
        statusItem.isVisible = true
        refresh()
    }

    func refresh() {
        let presentation = BirdPresentation(mode: store.mode)
        statusItem.button?.title = ""
        statusItem.button?.image = BirdIconRenderer.image(for: store.mode)
        statusItem.button?.imagePosition = .imageOnly
        statusItem.button?.toolTip = "不休眠猫头鹰 · \(presentation.statusTitle)"
    }

    @objc private func clicked(_ sender: Any?) {
        if NSApp.currentEvent?.type == .rightMouseUp {
            showMenu(from: nil, event: NSApp.currentEvent)
        } else {
            handle(.primary)
        }
    }

    private func handle(_ mouseButton: StatusBarMouseButton) {
        switch StatusBarInteraction.action(for: mouseButton) {
        case .openControlWindow: openWindow()
        case .showContextMenu: showMenu(from: nil, event: NSApp.currentEvent)
        }
    }

    private func showMenu(from sender: NSButton?, event: NSEvent?) {
        let p = BirdPresentation(mode: store.mode)
        let menu = NSMenu()
        let status = NSMenuItem(title: "\(p.emoji)  \(p.statusTitle)", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        if let snapshot = thermalMonitor.latestSnapshot {
            let thermal = ThermalPresentation(state: snapshot.thermalState)
            let top = snapshot.applications.first.map { " · \($0.usage.name) \(CPUUsageFormatter.string($0.usage.cpuPercent))" } ?? ""
            let item = NSMenuItem(title: "电脑状态：\(thermal.title)\(top)", action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        } else {
            let item = NSMenuItem(title: "电脑状态：正在获取应用占用", action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        menu.addItem(.separator())
        menu.addItem(item(p.toggleTitle, #selector(toggleMode)))
        menu.addItem(item("打开不休眠猫头鹰…", #selector(openControlWindow)))
        menu.addItem(.separator())
        let login = item("登录时自动启动", #selector(toggleLogin))
        login.state = launchController.isEnabled ? .on : .off
        menu.addItem(login)
        menu.addItem(.separator())
        menu.addItem(item("退出不休眠猫头鹰", #selector(quitApp)))
        if let sender, let event {
            NSMenu.popUpContextMenu(menu, with: event, for: sender)
        } else {
            statusItem.menu = menu
            statusItem.button?.performClick(nil)
            statusItem.menu = nil
        }
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
