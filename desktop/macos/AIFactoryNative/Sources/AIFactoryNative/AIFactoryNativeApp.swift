import SwiftUI

@main
struct AIFactoryNativeApp: App {
    @StateObject private var store = NativeWorkspaceStore()
    private let bridge = DesktopBridge()

    var body: some Scene {
        WindowGroup("AI-Factory", id: "main") {
            MainTabView(store: store, bridge: bridge)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("AI-Factory") {
                Button("Refresh Status") {
                    Task { await store.fetchStatus() }
                }
                .keyboardShortcut("r", modifiers: .command)
                Divider()
                Button("Open Workspace") {
                    bridge.open(store.dashboardURL)
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
                Button("Launch TUI") {
                    bridge.runShellCommand("python -m ai_factory.tui")
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            }
        }

        MenuBarExtra("AI-Factory", systemImage: store.apiReachable ? "cpu.fill" : "cpu") {
            MenuBarStatusView(store: store, bridge: bridge)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView()
        }
    }
}
