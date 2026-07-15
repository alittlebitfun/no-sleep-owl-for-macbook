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
