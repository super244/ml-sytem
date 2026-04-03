import SwiftUI

private let desktopLogDemoModeEnabled =
    ProcessInfo.processInfo.environment["AI_FACTORY_DESKTOP_DEMO_MODE"] == "1" ||
    ProcessInfo.processInfo.environment["AI_FACTORY_DEMO_MODE"] == "1"

struct LogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let level: String
    let message: String
    let source: String?
}

@MainActor
final class LogStreamStore: ObservableObject {
    @Published var entries: [LogEntry] = []
    @Published var isStreaming = false
    @Published var error: String? = nil
    @Published var filter: String = ""
    @Published var levelFilter: String = "ALL"

    private let apiURL: URL
    private var timer: Timer?
    private var cursor: String? = nil
    private let maxEntries = 500
    private let demoMode = desktopLogDemoModeEnabled

    init(apiURL: URL) {
        self.apiURL = apiURL
    }

    func startStreaming() {
        isStreaming = true
        tick()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    func stopStreaming() {
        isStreaming = false
        timer?.invalidate()
        timer = nil
    }

    func clear() {
        entries.removeAll()
        cursor = nil
    }

    private func tick() {
        Task { await fetchLogs() }
    }

    func fetchLogs() async {
        var components = URLComponents(url: apiURL.appendingPathComponent("v1/agents/logs"), resolvingAgainstBaseURL: false)!
        let queryItems: [URLQueryItem] = [URLQueryItem(name: "limit", value: "50")]
        components.queryItems = queryItems

        guard let url = components.url else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 4

        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                error = "HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)"
                if demoMode { seedSimulated() }
                return
            }
            let decoded = try JSONDecoder().decode(APILogsResponse.self, from: data)
            let newEntries = decoded.entries.map { raw in
                LogEntry(
                    timestamp: raw.timestampAsDate,
                    level: raw.level,
                    message: raw.message,
                    source: raw.source
                )
            }
            entries.append(contentsOf: newEntries)
            if entries.count > maxEntries { entries.removeFirst(entries.count - maxEntries) }
            cursor = nil
            error = nil
        } catch {
            self.error = error.localizedDescription
            if demoMode { seedSimulated() }
        }
    }

    private func seedSimulated() {
        let levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
        let messages = [
            "Training step completed",
            "Checkpoint saved to artifacts/",
            "GPU memory: 14.2 GB / 24 GB",
            "Validation loss: 0.3421",
            "Instance heartbeat received",
            "Scheduler tick — 0 pending tasks",
            "Model loaded from cache",
            "Request processed in 42ms",
        ]
        let entry = LogEntry(
            timestamp: Date(),
            level: levels.randomElement()!,
            message: messages.randomElement()!,
            source: "simulator"
        )
        entries.append(entry)
        if entries.count > maxEntries { entries.removeFirst() }
    }

    var filteredEntries: [LogEntry] {
        entries.filter { entry in
            let levelOK = levelFilter == "ALL" || entry.level == levelFilter
            let textOK = filter.isEmpty ||
                entry.message.localizedCaseInsensitiveContains(filter) ||
                (entry.source ?? "").localizedCaseInsensitiveContains(filter)
            return levelOK && textOK
        }
    }
}

struct APILogsResponse: Decodable {
    struct RawEntry: Decodable {
        let timestamp: Double?
        let level: String
        let message: String
        let source: String?

        var timestampAsDate: Date {
            if let timestamp {
                return Date(timeIntervalSince1970: timestamp)
            }
            return Date()
        }
    }
    let logs: [RawEntry]

    var entries: [RawEntry] { logs }
    let next_cursor: String? = nil
}

struct LogStreamView: View {
    @StateObject private var store: LogStreamStore
    @State private var autoScroll = true

    init(apiURL: URL) {
        _store = StateObject(wrappedValue: LogStreamStore(apiURL: apiURL))
    }

    private let levelColors: [String: Color] = [
        "INFO": .blue,
        "DEBUG": .secondary,
        "WARNING": .orange,
        "ERROR": .red,
        "CRITICAL": .red,
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            toolbar
            Divider()
            logList
            Divider()
            statusBar
        }
        .onAppear { store.startStreaming() }
        .onDisappear { store.stopStreaming() }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Log Stream")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("\(store.filteredEntries.count) entries")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            TextField("Filter…", text: $store.filter)
                .textFieldStyle(.roundedBorder)
                .frame(width: 180)
            Picker("Level", selection: $store.levelFilter) {
                ForEach(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"], id: \.self) { l in
                    Text(l).tag(l)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 100)
            Toggle("Auto-scroll", isOn: $autoScroll)
                .toggleStyle(.checkbox)
                .font(.system(size: 12, design: .rounded))
            Button {
                store.isStreaming ? store.stopStreaming() : store.startStreaming()
            } label: {
                Image(systemName: store.isStreaming ? "pause.fill" : "play.fill")
            }
            .buttonStyle(.bordered)
            .help(store.isStreaming ? "Pause stream" : "Resume stream")
            Button {
                store.clear()
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
            .help("Clear log")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    private var logList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(store.filteredEntries) { entry in
                        logRow(entry)
                            .id(entry.id)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .onChange(of: store.filteredEntries.count) { _, _ in
                if autoScroll, let last = store.filteredEntries.last {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func logRow(_ entry: LogEntry) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(entry.timestamp, format: .dateTime.hour().minute().second())
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.tertiary)
                .frame(width: 80, alignment: .leading)
            Text(entry.level)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(levelColors[entry.level] ?? .secondary)
                .frame(width: 56, alignment: .leading)
            if let src = entry.source {
                Text("[\(src)]")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Text(entry.message)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .background(
            entry.level == "ERROR" || entry.level == "CRITICAL"
                ? Color.red.opacity(0.06)
                : Color.clear
        )
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(store.isStreaming ? Color.green : Color.secondary)
                .frame(width: 7, height: 7)
            Text(store.isStreaming ? "Streaming" : "Paused")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
            if let err = store.error {
                Text("· \(err)")
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(.red.opacity(0.8))
            }
            Spacer()
            Text("\(store.entries.count) / 500 buffered")
                .font(.system(size: 11, design: .rounded))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
