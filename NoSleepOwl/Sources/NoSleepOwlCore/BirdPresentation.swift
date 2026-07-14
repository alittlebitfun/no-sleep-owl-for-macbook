import Foundation

public struct BirdPresentation: Equatable, Sendable {
    public let emoji: String
    public let statusTitle: String
    public let detail: String
    public let toggleTitle: String

    public init(mode: BirdMode) {
        switch mode {
        case .bird:
            emoji = "🐦"
            statusTitle = "小鸟可以休息"
            detail = "Mac 将按系统设置正常休眠"
            toggleTitle = "切换到猫头鹰模式"
        case .owl:
            emoji = "🦉"
            statusTitle = "猫头鹰正在守夜"
            detail = "屏幕可以熄灭，后台任务会继续运行"
            toggleTitle = "切换到小鸟模式"
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
