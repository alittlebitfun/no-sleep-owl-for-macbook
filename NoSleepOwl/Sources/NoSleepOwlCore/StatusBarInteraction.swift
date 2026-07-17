public enum StatusBarMouseButton {
    case primary
    case secondary
}

public enum StatusBarAction: Equatable {
    case openControlWindow
    case showContextMenu
}

public enum StatusBarInteraction {
    public static func action(for button: StatusBarMouseButton) -> StatusBarAction {
        switch button {
        case .primary: .openControlWindow
        case .secondary: .showContextMenu
        }
    }
}

public enum StatusItemPlacementPolicy {
    public static let usesFloatingOverlay = false
    public static let persistsCustomPosition = false
}

public enum ApplicationReopenPolicy {
    public static let opensControlWindow = true
}

public enum DisplayLocationPolicy {
    public static func activationPolicy(showsDockIcon: Bool) -> String {
        showsDockIcon ? "regular" : "accessory"
    }

    public static let keepsProcessAliveWhenHidden = true
}

public enum StatusMenuPolicy {
    public static let includesSettings = true
}
