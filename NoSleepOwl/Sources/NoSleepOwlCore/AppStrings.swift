import Foundation

public struct AppStrings: Sendable {
    public let language: AppLanguage
    public init(language: AppLanguage) { self.language = language }

    private func value(_ zh: String, _ en: String) -> String { language == .zhHans ? zh : en }

    public var appName: String { value("不休眠猫头鹰", "No Sleep Owl") }
    public var settingsMenuTitle: String { value("设置…", "Settings…") }
    public var settingsWindowTitle: String { value("不休眠猫头鹰设置", "No Sleep Owl Settings") }
    public var languageLabel: String { value("语言", "Language") }
    public var simplifiedChinese: String { "简体中文" }
    public var english: String { "English" }
    public var showThermalStatus: String { value("显示电脑热状态", "Show thermal status") }
    public var showHighUsageApps: String { value("显示高占用应用", "Show high-usage applications") }
    public var showStatusBarIcon: String { value("显示状态栏图标", "Show menu bar icon") }
    public var showDockIcon: String { value("显示 Dock 图标", "Show Dock icon") }
    public var loginAtStartup: String { value("登录时自动启动", "Launch at Login") }
    public var allowBattery: String { value("使用电池时也允许合盖守夜", "Allow closed-lid watch on battery") }
    public var installHelper: String { value("安装 / 批准辅助程序", "Install / Approve Helper") }
    public var openApplication: String { value("打开不休眠猫头鹰…", "Open No Sleep Owl…") }
    public var quitApplication: String { value("退出不休眠猫头鹰", "Quit No Sleep Owl") }
    public var computerStatusChecking: String { value("电脑状态：正在获取应用占用", "Computer status: Checking application usage") }
    public var monitoringCheckingTitle: String { value("电脑状态 · 正在检查", "Computer Status · Checking") }
    public var applicationUsageTitle: String { value("应用占用", "Application Usage") }
    public var fetchingUsage: String { value("正在获取应用占用", "Checking application usage") }
    public var firstSample: String { value("首次采样约需 10 秒", "The first sample takes about 10 seconds") }
    public var noSignificantUsage: String { value("暂未发现明显 CPU 占用", "No significant CPU usage detected") }
    public var terminate: String { value("退出", "Quit") }
    public var notWatching: String { value("尚未开始守夜", "Watch has not started") }
    public var helperApproved: String { value("辅助程序已批准 · 日常切换无需密码", "Helper approved · No password needed for daily switching") }
    public var helperNeedsApproval: String { value("等待在系统设置中批准", "Waiting for approval in System Settings") }
    public var helperNotRegistered: String { value("辅助程序尚未注册", "Helper is not registered") }
    public var helperNotFound: String { value("安装包中未找到辅助程序", "Helper was not found in the application bundle") }
    public var helperUnknown: String { value("辅助程序状态未知", "Helper status is unknown") }
    public var loginChangeFailed: String { value("未能修改登录启动设置。", "Could not change the launch-at-login setting.") }
    public var helperReady: String { value("辅助程序已准备好。", "Helper is ready.") }
    public var lowBatteryWarning: String { value("电量已降至 20%，建议接通电源。", "Battery reached 20%. Please connect power.") }
    public var thermalWarning: String { value("Mac 温度压力较高，请改善通风。", "Mac thermal pressure is high. Please improve ventilation.") }
    public var powerDisconnected: String { value("电源已断开，已按安全策略恢复小鸟模式。", "Power was disconnected. Bird mode was restored for safety.") }
    public var criticalBattery: String { value("电量低于 10%，已自动恢复小鸟模式。", "Battery is below 10%. Bird mode was restored automatically.") }
    public var thermalCritical: String { value("系统热状态危急，已自动恢复小鸟模式。", "Thermal state is critical. Bird mode was restored automatically.") }
    public var thermalNotificationSerious: String { value("Mac 温度压力较高", "Mac Thermal Pressure Is High") }
    public var thermalNotificationCritical: String { value("Mac 热状态危急", "Mac Thermal State Is Critical") }
    public var improveVentilation: String { value("请改善通风并检查正在运行的应用。", "Please improve ventilation and review running applications.") }

    public func duration(_ seconds: TimeInterval) -> String {
        value("已守夜  \(OwlDurationFormatter.string(seconds: seconds))", "Watching for  \(OwlDurationFormatter.string(seconds: seconds))")
    }

    public func computerStatus(_ title: String, top: String) -> String {
        value("电脑状态：\(title)\(top)", "Computer status: \(title)\(top)")
    }

    public func monitoringTitle(_ title: String) -> String {
        value("电脑状态 · \(title)", "Computer Status · \(title)")
    }

    public func helperUnavailable(_ message: String) -> String { value("辅助程序不可用：\(message)", "Helper unavailable: \(message)") }
    public func helperRequestFailed(_ message: String) -> String { value("辅助程序操作失败：\(message)", "Helper request failed: \(message)") }
    public func highUsageNotification(app: String, cpu: String) -> String {
        value("\(app) 当前占用约 \(cpu)，请检查通风和运行任务。", "\(app) is using about \(cpu). Please check ventilation and running tasks.")
    }
}
