import AppKit
import Foundation

final class DesktopBridge {
    func open(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

struct NativeAction: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let command: String
}

final class NativeWorkspaceStore: ObservableObject {
    @Published var dashboardURL: URL
    @Published var apiURL: URL
    @Published var artifactsURL: URL
    @Published var shellSummary: String

    init(processInfo: ProcessInfo = .processInfo) {
        let environment = processInfo.environment
        self.dashboardURL = URL(string: environment["AI_FACTORY_DESKTOP_URL"] ?? "http://127.0.0.1:3000/workspace")!
        self.apiURL = URL(string: environment["AI_FACTORY_API_URL"] ?? "http://127.0.0.1:8000")!
        self.artifactsURL = URL(fileURLWithPath: environment["AI_FACTORY_ARTIFACTS_DIR"] ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("ai-factory/artifacts").path)
        self.shellSummary = environment["AI_FACTORY_DESKTOP_SUMMARY"] ?? "Native SwiftUI shell with direct access to the shared control center."
    }

    var quickActions: [NativeAction] {
        [
            NativeAction(
                title: "Open workspace",
                detail: "Jump straight to the browser-backed control center.",
                command: dashboardURL.absoluteString
            ),
            NativeAction(
                title: "Open API root",
                detail: "Inspect the local FastAPI backend.",
                command: apiURL.absoluteString
            ),
            NativeAction(
                title: "Reveal artifacts",
                detail: "Open the shared artifacts directory in Finder.",
                command: artifactsURL.path
            ),
        ]
    }

    var launchCommand: String {
        "AI_FACTORY_DESKTOP_URL=\(dashboardURL.absoluteString) AI_FACTORY_API_URL=\(apiURL.absoluteString) swift run"
    }
}
