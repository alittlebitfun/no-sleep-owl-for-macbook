import AppKit

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
let delegateLifetime = Unmanaged.passRetained(delegate)
app.run()
delegateLifetime.release()
