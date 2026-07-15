import AppKit
import NoSleepOwlCore

private final class FlippedStackView: NSStackView {
    override var isFlipped: Bool { true }
}

@MainActor
final class ThermalStatusView: NSView {
    private let titleLabel = NSTextField(labelWithString: "电脑状态 · 正在检查")
    private let detailLabel = NSTextField(labelWithString: "正在获取应用占用")
    private let rows = FlippedStackView()
    private let scrollView = NSScrollView()
    var onTerminate: ((pid_t) -> Void)?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.cornerRadius = 14
        layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.72).cgColor

        titleLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        detailLabel.font = .systemFont(ofSize: 12)
        detailLabel.textColor = .secondaryLabelColor
        rows.orientation = .vertical
        rows.spacing = 8
        rows.alignment = .leading
        scrollView.documentView = rows
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.heightAnchor.constraint(equalToConstant: 128).isActive = true
        scrollView.widthAnchor.constraint(equalToConstant: 358).isActive = true

        let stack = NSStackView(views: [titleLabel, detailLabel, scrollView])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor),
            stack.topAnchor.constraint(equalTo: topAnchor),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor),
            widthAnchor.constraint(equalToConstant: 390)
        ])
    }

    required init?(coder: NSCoder) { nil }

    func update(snapshot: ThermalAppSnapshot?) {
        rows.arrangedSubviews.forEach { $0.removeFromSuperview() }
        guard let snapshot else {
            titleLabel.stringValue = "电脑状态 · 正在检查"
            detailLabel.stringValue = "正在获取应用占用"
            rows.addArrangedSubview(NSTextField(labelWithString: "首次采样约需 10 秒"))
            layoutRows()
            return
        }
        let presentation = ThermalPresentation(state: snapshot.thermalState)
        titleLabel.stringValue = "电脑状态 · \(presentation.title)"
        detailLabel.stringValue = presentation.detail
        titleLabel.textColor = color(for: snapshot.thermalState)
        if snapshot.applications.isEmpty {
            rows.addArrangedSubview(NSTextField(labelWithString: "暂未发现明显 CPU 占用"))
        } else {
            snapshot.applications.forEach { rows.addArrangedSubview(applicationRow($0)) }
        }
        layoutRows()
    }

    private func layoutRows() {
        rows.layoutSubtreeIfNeeded()
        rows.frame = NSRect(
            x: 0,
            y: 0,
            width: scrollView.contentSize.width,
            height: max(scrollView.contentSize.height, rows.fittingSize.height)
        )
    }

    private func applicationRow(_ app: MonitoredApplication) -> NSView {
        let icon = NSImageView(image: app.icon ?? NSImage(systemSymbolName: "app", accessibilityDescription: nil)!)
        icon.imageScaling = .scaleProportionallyUpOrDown
        icon.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([icon.widthAnchor.constraint(equalToConstant: 22), icon.heightAnchor.constraint(equalToConstant: 22)])

        let name = NSTextField(labelWithString: app.usage.name)
        name.lineBreakMode = .byTruncatingTail
        name.font = .systemFont(ofSize: 13, weight: app.usage.isSustainedHigh ? .semibold : .regular)
        if app.usage.isSustainedHigh { name.textColor = .systemRed }
        let cpu = NSTextField(labelWithString: CPUUsageFormatter.string(app.usage.cpuPercent))
        cpu.font = .monospacedDigitSystemFont(ofSize: 12, weight: .medium)
        cpu.textColor = app.usage.isSustainedHigh ? .systemRed : .secondaryLabelColor

        let row = NSStackView(views: [icon, name, cpu])
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        name.widthAnchor.constraint(equalToConstant: 190).isActive = true
        if app.usage.canTerminate {
            let button = NSButton(title: "退出", target: self, action: #selector(terminate(_:)))
            button.bezelStyle = .rounded
            button.controlSize = .small
            button.tag = Int(app.usage.pid)
            row.addArrangedSubview(button)
        }
        return row
    }

    @objc private func terminate(_ sender: NSButton) { onTerminate?(pid_t(sender.tag)) }

    private func color(for state: OwlThermalState) -> NSColor {
        switch state {
        case .nominal: .systemGreen
        case .fair: .systemOrange
        case .serious, .critical: .systemRed
        }
    }
}
