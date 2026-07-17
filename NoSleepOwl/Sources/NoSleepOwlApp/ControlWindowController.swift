import AppKit
import NoSleepOwlCore

@MainActor
final class ControlWindowController: NSObject, NSWindowDelegate {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let sleepController: PrivilegedSleepController
    private let safetyMonitor: SafetyMonitor
    private let thermalMonitor: ThermalAppMonitor
    private let preferences: AppPreferences
    private let window: NSWindow
    private let emoji = NSTextField(labelWithString: "")
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(labelWithString: "")
    private let durationLabel = NSTextField(labelWithString: "")
    private let errorLabel = NSTextField(labelWithString: "")
    private let toggleButton = NSButton(title: "", target: nil, action: nil)
    private let loginButton = NSButton(checkboxWithTitle: "登录时自动启动", target: nil, action: nil)
    private let batteryButton = NSButton(checkboxWithTitle: "使用电池时也允许合盖守夜", target: nil, action: nil)
    private let helperLabel = NSTextField(labelWithString: "")
    private let helperButton = NSButton(title: "安装 / 批准辅助程序", target: nil, action: nil)
    private let settingsButton = NSButton(title: "设置…", target: nil, action: nil)
    var onOpenSettings: (() -> Void)?
    var onVisibilityChange: ((Bool) -> Void)?
    private let thermalView = ThermalStatusView()
    private var timer: Timer?

    init(store: OwlModeStore, launchController: LaunchAtLoginController, sleepController: PrivilegedSleepController, safetyMonitor: SafetyMonitor, thermalMonitor: ThermalAppMonitor, preferences: AppPreferences) {
        self.store = store
        self.launchController = launchController
        self.sleepController = sleepController
        self.safetyMonitor = safetyMonitor
        self.thermalMonitor = thermalMonitor
        self.preferences = preferences
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 480, height: 760), styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        super.init()
        window.title = "不休眠猫头鹰"
        window.isReleasedWhenClosed = false
        window.delegate = self
        buildUI()
        refresh()
    }

    func show() {
        NSApp.activate(ignoringOtherApps: true)
        window.center()
        window.makeKeyAndOrderFront(nil)
        onVisibilityChange?(true)
        thermalMonitor.setWindowVisible(true)
        startTimer()
    }

    func refresh() {
        let snapshot = preferences.snapshot
        let strings = AppStrings(language: snapshot.language)
        let p = BirdPresentation(mode: store.mode, language: snapshot.language)
        window.title = strings.appName
        emoji.stringValue = p.emoji
        titleLabel.stringValue = p.statusTitle
        detailLabel.stringValue = p.detail
        toggleButton.title = p.toggleTitle
        errorLabel.stringValue = store.errorMessage ?? ""
        loginButton.title = strings.loginAtStartup
        loginButton.state = launchController.isEnabled ? .on : .off
        batteryButton.title = strings.allowBattery
        batteryButton.state = safetyMonitor.powerPolicy == .allowBattery ? .on : .off
        helperLabel.stringValue = sleepController.statusText(language: snapshot.language)
        helperButton.title = strings.installHelper
        settingsButton.title = strings.settingsMenuTitle
        helperButton.isHidden = sleepController.isReady
        let mode = MonitoringDisplayPolicy.mode(thermal: snapshot.showsThermalStatus, applications: snapshot.showsHighUsageApps)
        thermalView.isHidden = mode == .hidden
        thermalView.update(snapshot: thermalMonitor.latestSnapshot, mode: mode, strings: strings)
        updateDuration()
    }

    func windowWillClose(_ notification: Notification) {
        timer?.invalidate()
        timer = nil
        thermalMonitor.setWindowVisible(false)
        onVisibilityChange?(false)
    }

    private func buildUI() {
        emoji.font = .systemFont(ofSize: 92)
        emoji.alignment = .center
        titleLabel.font = .systemFont(ofSize: 26, weight: .bold)
        titleLabel.alignment = .center
        detailLabel.font = .systemFont(ofSize: 14)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.alignment = .center
        durationLabel.font = .monospacedDigitSystemFont(ofSize: 18, weight: .medium)
        durationLabel.alignment = .center
        errorLabel.textColor = .systemRed
        errorLabel.alignment = .center
        toggleButton.bezelStyle = .rounded
        toggleButton.controlSize = .large
        toggleButton.target = self
        toggleButton.action = #selector(toggleMode)
        loginButton.target = self
        loginButton.action = #selector(toggleLogin)
        batteryButton.target = self
        batteryButton.action = #selector(toggleBatteryPolicy)
        helperLabel.textColor = .secondaryLabelColor
        helperLabel.alignment = .center
        helperButton.target = self
        helperButton.action = #selector(registerHelper)
        settingsButton.target = self
        settingsButton.action = #selector(openSettings)

        thermalView.onTerminate = { [weak thermalMonitor] pid in thermalMonitor?.terminate(pid: pid) }
        let stack = NSStackView(views: [settingsButton, emoji, titleLabel, detailLabel, durationLabel, toggleButton, thermalView, helperLabel, helperButton, batteryButton, loginButton, errorLabel])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 18
        stack.edgeInsets = NSEdgeInsets(top: 30, left: 34, bottom: 26, right: 34)
        stack.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = NSView()
        window.contentView?.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor),
            stack.topAnchor.constraint(equalTo: window.contentView!.topAnchor),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: window.contentView!.bottomAnchor),
            detailLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 390),
            toggleButton.widthAnchor.constraint(equalToConstant: 260),
            toggleButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateDuration() }
        }
    }

    private func updateDuration() {
        if let startedAt = store.startedAt {
            durationLabel.stringValue = AppStrings(language: preferences.snapshot.language).duration(Date().timeIntervalSince(startedAt))
        } else { durationLabel.stringValue = AppStrings(language: preferences.snapshot.language).notWatching }
    }

    @objc private func toggleMode() { store.toggle() }
    @objc private func toggleLogin() {
        do { try launchController.setEnabled(loginButton.state == .on) }
        catch { errorLabel.stringValue = AppStrings(language: preferences.snapshot.language).loginChangeFailed }
        loginButton.state = launchController.isEnabled ? .on : .off
    }

    @objc private func toggleBatteryPolicy() {
        safetyMonitor.powerPolicy = batteryButton.state == .on ? .allowBattery : .acOnly
        safetyMonitor.evaluate()
    }

    @objc private func registerHelper() {
        let strings = AppStrings(language: preferences.snapshot.language)
        do { try sleepController.register(); store.setMessage(strings.helperReady) }
        catch { store.setMessage(strings.helperRequestFailed(error.localizedDescription)) }
        refresh()
    }

    @objc private func openSettings() { onOpenSettings?() }
}
