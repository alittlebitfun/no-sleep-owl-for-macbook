import AppKit
import NoSleepOwlCore

enum BirdIconRenderer {
    static func image(for mode: BirdMode) -> NSImage {
        let emoji = BirdPresentation(mode: mode).emoji as NSString
        let image = NSImage(size: NSSize(width: 18, height: 18), flipped: false) { rect in
            let attributes: [NSAttributedString.Key: Any] = [
                .font: NSFont(name: "Apple Color Emoji", size: 14) ?? NSFont.systemFont(ofSize: 14)
            ]
            let size = emoji.size(withAttributes: attributes)
            emoji.draw(at: NSPoint(x: (rect.width - size.width) / 2, y: (rect.height - size.height) / 2), withAttributes: attributes)
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = mode == .owl ? "猫头鹰模式" : "小鸟模式"
        return image
    }
}
