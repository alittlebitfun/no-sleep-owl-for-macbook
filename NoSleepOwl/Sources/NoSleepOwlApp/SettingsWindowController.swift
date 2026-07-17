import AppKit
import NoSleepOwlCore

@MainActor
final class SettingsWindowController: NSObject, NSWindowDelegate {
    private let preferences: AppPreferences
    private let window: NSWindow
    private let languageLabel = NSTextField(labelWithString: "")
    private let languagePopup = NSPopUpButton()
    private let thermalButton = NSButton(checkboxWithTitle: "", target: nil, action: nil)
    private let applicationsButton = NSButton(checkboxWithTitle: "", target: nil, action: nil)
    private let statusBarButton = NSButton(checkboxWithTitle: "", target: nil, action: nil)
    private let dockButton = NSButton(checkboxWithTitle: "", target: nil, action: nil)

    init(preferences: AppPreferences) {
        self.preferences = preferences
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 250),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        super.init()
        window.isReleasedWhenClosed = false
        window.delegate = self
        buildUI()
        refresh()
    }

    func show() {
        NSApp.activate(ignoringOtherApps: true)
        if !window.isVisible { window.center() }
        window.makeKeyAndOrderFront(nil)
    }

    func refresh() {
        let snapshot = preferences.snapshot
        let strings = AppStrings(language: snapshot.language)
        window.title = strings.settingsWindowTitle
        languageLabel.stringValue = strings.languageLabel
        languagePopup.removeAllItems()
        languagePopup.addItems(withTitles: [strings.simplifiedChinese, strings.english])
        languagePopup.selectItem(at: snapshot.language == .zhHans ? 0 : 1)
        thermalButton.title = strings.showThermalStatus
        thermalButton.state = snapshot.showsThermalStatus ? .on : .off
        applicationsButton.title = strings.showHighUsageApps
        applicationsButton.state = snapshot.showsHighUsageApps ? .on : .off
        statusBarButton.title = strings.showStatusBarIcon
        statusBarButton.state = snapshot.showsStatusBarIcon ? .on : .off
        dockButton.title = strings.showDockIcon
        dockButton.state = snapshot.showsDockIcon ? .on : .off
    }

    private func buildUI() {
        languageLabel.font = .systemFont(ofSize: 13, weight: .medium)
        languagePopup.target = self
        languagePopup.action = #selector(changeLanguage)
        thermalButton.target = self
        thermalButton.action = #selector(changeThermal)
        applicationsButton.target = self
        applicationsButton.action = #selector(changeApplications)
        statusBarButton.target = self
        statusBarButton.action = #selector(changeStatusBar)
        dockButton.target = self
        dockButton.action = #selector(changeDock)

        let languageRow = NSStackView(views: [languageLabel, languagePopup])
        languageRow.orientation = .horizontal
        languageRow.alignment = .centerY
        languageRow.distribution = .fillEqually
        languageRow.spacing = 20

        let separator = NSBox()
        separator.boxType = .separator
        let stack = NSStackView(views: [languageRow, separator, thermalButton, applicationsButton, statusBarButton, dockButton])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 22
        stack.edgeInsets = NSEdgeInsets(top: 30, left: 32, bottom: 30, right: 32)
        stack.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = NSView()
        window.contentView?.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor),
            stack.topAnchor.constraint(equalTo: window.contentView!.topAnchor),
            languageRow.widthAnchor.constraint(equalToConstant: 356)
        ])
    }

    @objc private func changeLanguage() { preferences.setLanguage(languagePopup.indexOfSelectedItem == 0 ? .zhHans : .en) }
    @objc private func changeThermal() { preferences.setShowsThermalStatus(thermalButton.state == .on) }
    @objc private func changeApplications() { preferences.setShowsHighUsageApps(applicationsButton.state == .on) }
    @objc private func changeStatusBar() { preferences.setShowsStatusBarIcon(statusBarButton.state == .on) }
    @objc private func changeDock() { preferences.setShowsDockIcon(dockButton.state == .on) }
}
