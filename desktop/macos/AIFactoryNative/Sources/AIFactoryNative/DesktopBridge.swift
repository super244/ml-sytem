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

    func runShellCommand(_ command: String) {
        let task = Process()
        task.launchPath = "/bin/zsh"
        task.arguments = ["-c", command]
        try? task.run()
    }
}

struct NativeAction: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let command: String
    let icon: String
}

struct APIStatusResponse: Decodable {
    let instance_count: Int?
    let running_count: Int?
    let version: String?
    let uptime_seconds: Double?
}

final class NativeWorkspaceStore: ObservableObject {
    @Published var dashboardURL: URL
    @Published var apiURL: URL
    @Published var artifactsURL: URL
    @Published var shellSummary: String

    @Published var apiReachable: Bool = false
    @Published var instanceCount: Int = 0
    @Published var runningCount: Int = 0
    @Published var apiVersion: String = "—"
    @Published var uptimeSeconds: Double = 0
    @Published var lastChecked: Date? = nil
    @Published var statusError: String? = nil

    private var pollTimer: Timer?

    init(processInfo: ProcessInfo = .processInfo) {
        let environment = processInfo.environment
        self.dashboardURL = URL(string: environment["AI_FACTORY_DESKTOP_URL"] ?? "http://127.0.0.1:3000/workspace")!
        self.apiURL = URL(string: environment["AI_FACTORY_API_URL"] ?? "http://127.0.0.1:8000")!
        self.artifactsURL = URL(fileURLWithPath: environment["AI_FACTORY_ARTIFACTS_DIR"] ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("ai-factory/artifacts").path)
        self.shellSummary = environment["AI_FACTORY_DESKTOP_SUMMARY"] ?? "Native SwiftUI shell with direct access to the shared control center."
        startPolling()
    }

    deinit {
        pollTimer?.invalidate()
    }

    func startPolling() {
        fetchStatusBackground()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 8.0, repeats: true) { [weak self] _ in
            self?.fetchStatusBackground()
        }
    }

    private func fetchStatusBackground() {
        Task { await fetchStatus() }
    }

    @MainActor
    func fetchStatus() async {
        let endpoint = apiURL.appendingPathComponent("v1/status")
        var request = URLRequest(url: endpoint)
        request.timeoutInterval = 5
        request.httpMethod = "GET"

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                apiReachable = false
                statusError = "HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"
                lastChecked = Date()
                return
            }
            let status = try JSONDecoder().decode(APIStatusResponse.self, from: data)
            apiReachable = true
            instanceCount = status.instance_count ?? 0
            runningCount = status.running_count ?? 0
            apiVersion = status.version ?? "—"
            uptimeSeconds = status.uptime_seconds ?? 0
            statusError = nil
            lastChecked = Date()
        } catch {
            apiReachable = false
            statusError = error.localizedDescription
            lastChecked = Date()
        }
    }

    var quickActions: [NativeAction] {
        [
            NativeAction(
                title: "Open workspace",
                detail: "Jump straight to the browser-backed control center.",
                command: dashboardURL.absoluteString,
                icon: "globe"
            ),
            NativeAction(
                title: "Open API root",
                detail: "Inspect the local FastAPI backend.",
                command: apiURL.absoluteString,
                icon: "server.rack"
            ),
            NativeAction(
                title: "Reveal artifacts",
                detail: "Open the shared artifacts directory in Finder.",
                command: artifactsURL.path,
                icon: "folder"
            ),
            NativeAction(
                title: "Launch TUI",
                detail: "Open the terminal-based mission console.",
                command: "python -m ai_factory.tui",
                icon: "terminal"
            ),
        ]
    }

    var launchCommand: String {
        "AI_FACTORY_DESKTOP_URL=\(dashboardURL.absoluteString) AI_FACTORY_API_URL=\(apiURL.absoluteString) swift run"
    }

    var formattedUptime: String {
        let h = Int(uptimeSeconds) / 3600
        let m = (Int(uptimeSeconds) % 3600) / 60
        let s = Int(uptimeSeconds) % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }
}
