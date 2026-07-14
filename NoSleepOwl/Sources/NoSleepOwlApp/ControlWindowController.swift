import AppKit
import NoSleepOwlCore

@MainActor
final class ControlWindowController: NSObject, NSWindowDelegate {
    private let store: OwlModeStore
    private let launchController: LaunchAtLoginController
    private let window: NSWindow
    private let emoji = NSTextField(labelWithString: "")
    private let titleLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(labelWithString: "")
    private let durationLabel = NSTextField(labelWithString: "")
    private let errorLabel = NSTextField(labelWithString: "")
    private let toggleButton = NSButton(title: "", target: nil, action: nil)
    private let loginButton = NSButton(checkboxWithTitle: "登录时自动启动", target: nil, action: nil)
    private var timer: Timer?

    init(store: OwlModeStore, launchController: LaunchAtLoginController) {
        self.store = store
        self.launchController = launchController
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 460, height: 470), styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
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
        startTimer()
    }

    func refresh() {
        let p = BirdPresentation(mode: store.mode)
        emoji.stringValue = p.emoji
        titleLabel.stringValue = p.statusTitle
        detailLabel.stringValue = p.detail
        toggleButton.title = p.toggleTitle
        errorLabel.stringValue = store.errorMessage ?? ""
        loginButton.state = launchController.isEnabled ? .on : .off
        updateDuration()
    }

    func windowWillClose(_ notification: Notification) { timer?.invalidate(); timer = nil }

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

        let stack = NSStackView(views: [emoji, titleLabel, detailLabel, durationLabel, toggleButton, loginButton, errorLabel])
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
            durationLabel.stringValue = "已守夜  \(OwlDurationFormatter.string(seconds: Date().timeIntervalSince(startedAt)))"
        } else { durationLabel.stringValue = "尚未开始守夜" }
    }

    @objc private func toggleMode() { store.toggle() }
    @objc private func toggleLogin() {
        do { try launchController.setEnabled(loginButton.state == .on) }
        catch { errorLabel.stringValue = "未能修改登录启动设置。" }
        loginButton.state = launchController.isEnabled ? .on : .off
    }
}
