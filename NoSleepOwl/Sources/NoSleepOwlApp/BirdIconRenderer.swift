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
        let image = baseImage.withSymbolConfiguration(
            NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        ) ?? baseImage
        image.isTemplate = true
        image.accessibilityDescription = description
        return image
    }
}
