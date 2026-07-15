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
    private var fallbackPanel: NSPanel?
    private weak var fallbackButton: StatusBarFallbackButton?

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
        statusItem.autosaveName = "NoSleepOwl.StatusItem.v2"
        statusItem.isVisible = true
        refresh()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.restoreClippedStatusItem()
        }
    }

    private func restoreClippedStatusItem() {
        guard let window = statusItem.button?.window else { return }
        let windows = CGWindowListCopyWindowInfo(.optionAll, kCGNullWindowID) as? [[String: Any]] ?? []
        let ownInfo = windows.first {
            ($0[kCGWindowNumber as String] as? NSNumber)?.intValue == window.windowNumber
        }
        if (ownInfo?[kCGWindowIsOnscreen as String] as? NSNumber)?.boolValue == true { return }
        guard let screen = window.screen ?? NSScreen.screens.first else { return }
        let x = max(screen.frame.minX + 300, screen.frame.maxX - 660)
        let y = screen.frame.maxY - window.frame.height
        showFallbackButton(frame: NSRect(x: x, y: y, width: window.frame.width, height: window.frame.height))
    }

    private func showFallbackButton(frame: NSRect) {
        let panel = NSPanel(
            contentRect: frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = NSWindow.Level(rawValue: NSWindow.Level.statusBar.rawValue + 1)
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let button = StatusBarFallbackButton(frame: NSRect(origin: .zero, size: frame.size))
        button.image = BirdIconRenderer.image(for: store.mode)
        button.toolTip = "不休眠猫头鹰"
        button.onPrimaryClick = { [weak self] in self?.handle(.primary) }
        button.onSecondaryClick = { [weak self, weak button] event in
            guard let self, let button else { return }
            self.showMenu(from: button, event: event)
        }
        panel.contentView = button
        panel.orderFrontRegardless()
        fallbackPanel = panel
        fallbackButton = button
    }

    func refresh() {
        let presentation = BirdPresentation(mode: store.mode)
        statusItem.button?.title = ""
        statusItem.button?.image = BirdIconRenderer.image(for: store.mode)
        statusItem.button?.imagePosition = .imageOnly
        statusItem.button?.toolTip = "不休眠猫头鹰 · \(presentation.statusTitle)"
        fallbackButton?.image = BirdIconRenderer.image(for: store.mode)
        fallbackButton?.toolTip = "不休眠猫头鹰 · \(presentation.statusTitle)"
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
        if sender === fallbackButton, let sender, let event {
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

@MainActor
private final class StatusBarFallbackButton: NSButton {
    var onPrimaryClick: (() -> Void)?
    var onSecondaryClick: ((NSEvent) -> Void)?
    private var trackingAreaReference: NSTrackingArea?
    private var isHovered = false

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        isBordered = false
        imagePosition = .imageOnly
        contentTintColor = .white
        focusRingType = .none
    }

    required init?(coder: NSCoder) { nil }

    override func mouseDown(with event: NSEvent) {
        onPrimaryClick?()
    }

    override func rightMouseDown(with event: NSEvent) {
        onSecondaryClick?(event)
    }

    override func updateTrackingAreas() {
        if let trackingAreaReference { removeTrackingArea(trackingAreaReference) }
        let tracking = NSTrackingArea(
            rect: bounds,
            options: [.activeAlways, .mouseEnteredAndExited, .inVisibleRect],
            owner: self
        )
        addTrackingArea(tracking)
        trackingAreaReference = tracking
        super.updateTrackingAreas()
    }

    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        needsDisplay = true
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        if isHovered {
            NSColor.white.withAlphaComponent(0.14).setFill()
            NSBezierPath(roundedRect: bounds.insetBy(dx: 4, dy: 4), xRadius: 7, yRadius: 7).fill()
        }
        super.draw(dirtyRect)
    }
}
