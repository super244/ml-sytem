// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "AIFactoryNative",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "AIFactoryNative", targets: ["AIFactoryNative"])
    ],
    targets: [
        .executableTarget(
            name: "AIFactoryNative",
            path: "Sources/AIFactoryNative",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        )
    ]
)
