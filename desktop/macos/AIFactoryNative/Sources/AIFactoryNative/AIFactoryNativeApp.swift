import SwiftUI

@main
struct AIFactoryNativeApp: App {
    @StateObject private var store = NativeWorkspaceStore()
    private let bridge = DesktopBridge()

    var body: some Scene {
        WindowGroup {
            ControlCenterView(store: store, bridge: bridge)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
    }
}
