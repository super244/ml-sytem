import SwiftUI

struct InstanceRecord: Identifiable, Decodable {
    let id: String
    let name: String
    let status: String
    let model: String?
    let created_at: String?
    let gpu: String?
    let task_count: Int?
}

struct APIInstancesResponse: Decodable {
    let instances: [InstanceRecord]
}

@MainActor
final class InstanceListStore: ObservableObject {
    @Published var instances: [InstanceRecord] = []
    @Published var isLoading = false
    @Published var error: String? = nil
    @Published var actionInFlight: String? = nil

    private let apiURL: URL
    private var timer: Timer?

    init(apiURL: URL) {
        self.apiURL = apiURL
    }

    func startPolling() {
        Task { await fetchInstances() }
        timer = Timer.scheduledTimer(withTimeInterval: 6.0, repeats: true) { [weak self] _ in
            Task { await self?.fetchInstances() }
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func fetchInstances() async {
        isLoading = true
        defer { isLoading = false }
        let endpoint = apiURL.appendingPathComponent("v1/instances")
        var req = URLRequest(url: endpoint)
        req.timeoutInterval = 5
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                error = "HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)"
                return
            }
            let decoded = try JSONDecoder().decode(APIInstancesResponse.self, from: data)
            instances = decoded.instances
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }

    func performAction(_ action: String, on instanceID: String) async {
        actionInFlight = instanceID
        defer { actionInFlight = nil }
        let endpoint = apiURL.appendingPathComponent("v1/instances/\(instanceID)/\(action)")
        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = 10
        do {
            let (_, _) = try await URLSession.shared.data(for: req)
            await fetchInstances()
        } catch {
            self.error = error.localizedDescription
        }
    }
}

struct InstanceListView: View {
    @StateObject private var store: InstanceListStore
    @State private var searchText = ""
    @State private var selectedID: String? = nil

    init(apiURL: URL) {
        _store = StateObject(wrappedValue: InstanceListStore(apiURL: apiURL))
    }

    private var filtered: [InstanceRecord] {
        guard !searchText.isEmpty else { return store.instances }
        return store.instances.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.status.localizedCaseInsensitiveContains(searchText) ||
            ($0.model ?? "").localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            toolbar
            Divider()
            if store.instances.isEmpty && !store.isLoading {
                emptyState
            } else {
                instanceTable
            }
            if let err = store.error {
                errorBanner(err)
            }
        }
        .onAppear { store.startPolling() }
        .onDisappear { store.stopPolling() }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Instances")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("\(store.instances.count) total · \(store.instances.filter { $0.status == "running" }.count) running")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            TextField("Search…", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(width: 200)
            Button {
                Task { await store.fetchInstances() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .help("Refresh instances")
            if store.isLoading {
                ProgressView().scaleEffect(0.7)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    private var instanceTable: some View {
        Table(filtered, selection: $selectedID) {
            TableColumn("Name") { rec in
                Text(rec.name)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
            }
            .width(min: 140, ideal: 200)

            TableColumn("Status") { rec in
                statusBadge(rec.status)
            }
            .width(90)

            TableColumn("Model") { rec in
                Text(rec.model ?? "—")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .width(min: 100, ideal: 160)

            TableColumn("GPU") { rec in
                Text(rec.gpu ?? "—")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            .width(80)

            TableColumn("Tasks") { rec in
                Text("\(rec.task_count ?? 0)")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            .width(50)

            TableColumn("Actions") { rec in
                HStack(spacing: 6) {
                    if rec.status == "running" {
                        actionButton("Stop", color: .red, instanceID: rec.id, action: "stop")
                    } else {
                        actionButton("Start", color: .green, instanceID: rec.id, action: "start")
                    }
                    actionButton("Delete", color: .secondary, instanceID: rec.id, action: "delete")
                }
            }
            .width(130)
        }
    }

    private func actionButton(_ label: String, color: Color, instanceID: String, action: String) -> some View {
        Button(label) {
            Task { await store.performAction(action, on: instanceID) }
        }
        .buttonStyle(.bordered)
        .tint(color)
        .disabled(store.actionInFlight == instanceID)
        .font(.system(size: 11, weight: .medium, design: .rounded))
    }

    private func statusBadge(_ status: String) -> some View {
        let color: Color = switch status {
        case "running": .green
        case "stopped": .secondary
        case "error": .red
        case "pending": .orange
        default: .secondary
        }
        return Text(status)
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.12), in: Capsule())
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "square.stack.3d.up.slash")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            Text("No instances found")
                .font(.system(size: 16, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
            Text("Start the backend and create an instance to see it here.")
                .font(.system(size: 13, design: .rounded))
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.system(size: 12, design: .rounded))
                .foregroundStyle(.red)
            Spacer()
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .background(Color.red.opacity(0.08))
    }
}
