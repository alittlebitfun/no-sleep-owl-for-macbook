import Foundation

guard geteuid() == 0 else {
    fputs("NoSleepOwlHelper must run as root\n", stderr)
    exit(77)
}

let service = HelperService()
let listener = NSXPCListener(machServiceName: "com.shiying.NoSleepOwl.helper")
let delegate = HelperListenerDelegate(service: service)
listener.delegate = delegate
service.startTimer()
listener.resume()
RunLoop.main.run()
