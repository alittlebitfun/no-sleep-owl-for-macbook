import AppKit
import NoSleepOwlCore

enum BirdIconRenderer {
    static func image(for mode: BirdMode, language: AppLanguage = .zhHans) -> NSImage {
        let symbolName = mode == .owl ? "moon.stars.fill" : "bird.fill"
        let description = language == .en
            ? (mode == .owl ? "Owl Mode" : "Bird Mode")
            : (mode == .owl ? "猫头鹰模式" : "小鸟模式")
        let baseImage = NSImage(systemSymbolName: symbolName, accessibilityDescription: description)
            ?? NSImage(systemSymbolName: "bird", accessibilityDescription: description)!
        let sizing = NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        let whitePalette = NSImage.SymbolConfiguration(paletteColors: [.white])
        let image = baseImage.withSymbolConfiguration(sizing.applying(whitePalette)) ?? baseImage
        // The menu bar uses a white symbol in both light and dark appearances.
        // A palette configuration avoids inheriting the app's default black tint.
        image.isTemplate = false
        image.accessibilityDescription = description
        return image
    }
}
