// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "NoSleepOwl",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "NoSleepOwlCore", targets: ["NoSleepOwlCore"]),
        .executable(name: "NoSleepOwlApp", targets: ["NoSleepOwlApp"]),
        .executable(name: "NoSleepOwlHelper", targets: ["NoSleepOwlHelper"])
    ],
    targets: [
        .target(name: "NoSleepOwlCore"),
        .executableTarget(name: "NoSleepOwlApp", dependencies: ["NoSleepOwlCore"]),
        .executableTarget(name: "NoSleepOwlHelper", dependencies: ["NoSleepOwlCore"]),
        .executableTarget(
            name: "NoSleepOwlTests",
            dependencies: ["NoSleepOwlCore"],
            path: "Tests/NoSleepOwlCoreTests"
        )
    ]
)
