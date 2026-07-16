import Foundation

public struct BirdPresentation: Equatable, Sendable {
    public let emoji: String
    public let statusTitle: String
    public let detail: String
    public let toggleTitle: String

    public init(mode: BirdMode, language: AppLanguage = .zhHans) {
        let en = language == .en
        switch mode {
        case .bird:
            emoji = "🐦"
            statusTitle = en ? "Bird can rest" : "小鸟可以休息"
            detail = en ? "Mac follows the system sleep settings" : "Mac 将按系统设置正常休眠"
            toggleTitle = en ? "Switch to Owl Mode" : "切换到猫头鹰模式"
        case .owl:
            emoji = "🦉"
            statusTitle = en ? "Owl is keeping watch" : "猫头鹰正在守夜"
            detail = en ? "Network and apps keep running with the lid closed · Keep the Mac ventilated" : "合盖后网络与应用继续运行 · 请保持通风，勿放入包中"
            toggleTitle = en ? "Switch to Bird Mode" : "切换到小鸟模式"
        }
    }
}

public enum OwlDurationFormatter {
    public static func string(seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let seconds = total % 60
        if hours > 0 { return String(format: "%02d:%02d:%02d", hours, minutes, seconds) }
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
