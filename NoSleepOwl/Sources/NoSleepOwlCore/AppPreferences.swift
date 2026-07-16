import Foundation

public enum AppLanguage: String, Sendable, CaseIterable {
    case zhHans
    case en
}

public struct AppPreferenceSnapshot: Sendable, Equatable {
    public let language: AppLanguage
    public let showsThermalStatus: Bool
    public let showsHighUsageApps: Bool

    public init(language: AppLanguage, showsThermalStatus: Bool, showsHighUsageApps: Bool) {
        self.language = language
        self.showsThermalStatus = showsThermalStatus
        self.showsHighUsageApps = showsHighUsageApps
    }
}

public final class AppPreferences {
    public enum Keys {
        public static let language = "language"
        public static let showsThermalStatus = "showsThermalStatus"
        public static let showsHighUsageApps = "showsHighUsageApps"
    }

    private let defaults: UserDefaults
    public private(set) var snapshot: AppPreferenceSnapshot
    public var onChange: (() -> Void)?

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let language = defaults.string(forKey: Keys.language).flatMap(AppLanguage.init(rawValue:)) ?? .zhHans
        let thermal = defaults.object(forKey: Keys.showsThermalStatus) as? Bool ?? true
        let applications = defaults.object(forKey: Keys.showsHighUsageApps) as? Bool ?? true
        snapshot = AppPreferenceSnapshot(language: language, showsThermalStatus: thermal, showsHighUsageApps: applications)
    }

    public func setLanguage(_ value: AppLanguage) {
        update(AppPreferenceSnapshot(language: value, showsThermalStatus: snapshot.showsThermalStatus, showsHighUsageApps: snapshot.showsHighUsageApps))
    }

    public func setShowsThermalStatus(_ value: Bool) {
        update(AppPreferenceSnapshot(language: snapshot.language, showsThermalStatus: value, showsHighUsageApps: snapshot.showsHighUsageApps))
    }

    public func setShowsHighUsageApps(_ value: Bool) {
        update(AppPreferenceSnapshot(language: snapshot.language, showsThermalStatus: snapshot.showsThermalStatus, showsHighUsageApps: value))
    }

    private func update(_ value: AppPreferenceSnapshot) {
        guard value != snapshot else { return }
        snapshot = value
        defaults.set(value.language.rawValue, forKey: Keys.language)
        defaults.set(value.showsThermalStatus, forKey: Keys.showsThermalStatus)
        defaults.set(value.showsHighUsageApps, forKey: Keys.showsHighUsageApps)
        onChange?()
    }
}

public enum MonitoringDisplayMode: Sendable, Equatable {
    case full
    case thermalOnly
    case applicationsOnly
    case hidden
}

public enum MonitoringDisplayPolicy {
    public static func mode(thermal: Bool, applications: Bool) -> MonitoringDisplayMode {
        switch (thermal, applications) {
        case (true, true): .full
        case (true, false): .thermalOnly
        case (false, true): .applicationsOnly
        case (false, false): .hidden
        }
    }
}

public enum MonitoringSamplingPolicy {
    public static func samplesApplications(showsHighUsageApps: Bool) -> Bool { showsHighUsageApps }
}
