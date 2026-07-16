#!/usr/bin/env swift

import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let sourceURL = root.appendingPathComponent("Resources/AppIconArtwork.png")
let outputURL = root.appendingPathComponent("Resources/AppIcon.png")

guard let source = NSImage(contentsOf: sourceURL) else {
    fatalError("Unable to read preserved artwork at \(sourceURL.path)")
}

let canvasSize = 1024
guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: canvasSize,
    pixelsHigh: canvasSize, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
    isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else {
    fatalError("Unable to create icon bitmap")
}

bitmap.size = NSSize(width: canvasSize, height: canvasSize)
NSGraphicsContext.saveGraphicsState()
guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
    fatalError("Unable to create drawing context")
}
NSGraphicsContext.current = context
context.imageInterpolation = .high
NSColor.clear.setFill()
let canvas = NSRect(x: 0, y: 0, width: canvasSize, height: canvasSize)
canvas.fill()
let cornerRadius = CGFloat(canvasSize) * 0.2237
NSBezierPath(roundedRect: canvas, xRadius: cornerRadius, yRadius: cornerRadius).addClip()
source.draw(in: canvas, from: .zero, operation: .sourceOver, fraction: 1)
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()

guard let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Unable to encode icon PNG")
}
try png.write(to: outputURL, options: .atomic)
