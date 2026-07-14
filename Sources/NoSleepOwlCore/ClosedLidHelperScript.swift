import Foundation

public enum ClosedLidHelperScript {
    public static func make(markerPath: String, appPID: Int32, restoreValue: Int) -> String {
        let marker = shellQuote(markerPath)
        return """
        cleanup() { /usr/bin/pmset -a disablesleep \(restoreValue); /bin/rm -f \(marker); }
        trap cleanup EXIT HUP INT TERM
        /usr/bin/pmset -a disablesleep 1
        while /bin/kill -0 \(appPID) 2>/dev/null && /bin/test -e \(marker); do /bin/sleep 1; done
        """
    }

    public static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    public static func launchCommand(script: String, logPath: String) -> String {
        "/bin/sh -c \(shellQuote(script)) >\(shellQuote(logPath)) 2>&1 </dev/null &"
    }
}
