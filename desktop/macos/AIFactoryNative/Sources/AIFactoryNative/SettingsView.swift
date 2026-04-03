import SwiftUI

struct SettingsView: View {
    @AppStorage("ai_factory_dashboard_url") private var dashboardURL = "http://127.0.0.1:3000/dashboard"
    @AppStorage("ai_factory_api_url") private var apiURL = "http://127.0.0.1:8000"
    @AppStorage("ai_factory_artifacts_dir") private var artifactsDir = ""
    @AppStorage("ai_factory_poll_interval") private var pollInterval = 8.0
    @AppStorage("ai_factory_theme") private var theme = "system"
    @AppStorage("ai_factory_notifications") private var notificationsEnabled = true
    @AppStorage("ai_factory_log_level") private var logLevel = "INFO"

    @State private var saved = false
    @State private var testResult: String? = nil
    @State private var isTesting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                header
                endpointsSection
                behaviourSection
                appearanceSection
                dangerSection
            }
            .padding(28)
        }
        .frame(minWidth: 640, minHeight: 560)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Settings")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("Configure endpoints, polling, and appearance.")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if saved {
                Label("Saved", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(.green)
                    .transition(.opacity)
            }
            Button("Save") {
                withAnimation { saved = true }
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    withAnimation { saved = false }
                }
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var endpointsSection: some View {
        settingsCard(title: "Endpoints", icon: "network") {
            VStack(alignment: .leading, spacing: 14) {
                labeledField("Dashboard URL", placeholder: "http://127.0.0.1:3000/dashboard", value: $dashboardURL)
                labeledField("API URL", placeholder: "http://127.0.0.1:8000", value: $apiURL)
                labeledField("Artifacts directory", placeholder: "~/ai-factory/artifacts", value: $artifactsDir)
                HStack(spacing: 10) {
                    Button("Test connection") {
                        Task { await testConnection() }
                    }
                    .buttonStyle(.bordered)
                    .disabled(isTesting)
                    if isTesting { ProgressView().scaleEffect(0.7) }
                    if let result = testResult {
                        Text(result)
                            .font(.system(size: 12, design: .rounded))
                            .foregroundStyle(result.hasPrefix("OK") ? .green : .red)
                    }
                }
            }
        }
    }

    private var behaviourSection: some View {
        settingsCard(title: "Behaviour", icon: "gearshape") {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Poll interval")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                        .frame(width: 160, alignment: .leading)
                    Slider(value: $pollInterval, in: 2...60, step: 1)
                        .frame(width: 200)
                    Text("\(Int(pollInterval))s")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 36)
                }
                HStack {
                    Text("Log level")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                        .frame(width: 160, alignment: .leading)
                    Picker("", selection: $logLevel) {
                        ForEach(["DEBUG", "INFO", "WARNING", "ERROR"], id: \.self) { l in
                            Text(l).tag(l)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 280)
                }
                Toggle(isOn: $notificationsEnabled) {
                    Text("Enable desktop notifications")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                }
                .toggleStyle(.switch)
            }
        }
    }

    private var appearanceSection: some View {
        settingsCard(title: "Appearance", icon: "paintbrush") {
            HStack {
                Text("Theme")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .frame(width: 160, alignment: .leading)
                Picker("", selection: $theme) {
                    Text("System").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .pickerStyle(.segmented)
                .frame(width: 240)
            }
        }
    }

    private var dangerSection: some View {
        settingsCard(title: "Reset", icon: "exclamationmark.triangle") {
            HStack(spacing: 12) {
                Button("Reset to defaults") {
                    dashboardURL = "http://127.0.0.1:3000/dashboard"
                    apiURL = "http://127.0.0.1:8000"
                    artifactsDir = ""
                    pollInterval = 8.0
                    theme = "system"
                    notificationsEnabled = true
                    logLevel = "INFO"
                }
                .buttonStyle(.bordered)
                .tint(.red)
                Text("Restores all settings to their factory defaults.")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func settingsCard<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: icon)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
            content()
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.black.opacity(0.06))
        )
    }

    private func labeledField(_ label: String, placeholder: String, value: Binding<String>) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .frame(width: 160, alignment: .leading)
            TextField(placeholder, text: value)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))
        }
    }

    private func testConnection() async {
        isTesting = true
        testResult = nil
        defer { isTesting = false }
        guard let url = URL(string: apiURL.isEmpty ? "http://127.0.0.1:8000" : apiURL) else {
            testResult = "Invalid URL"
            return
        }
        var req = URLRequest(url: url.appendingPathComponent("v1/status"))
        req.timeoutInterval = 5
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            testResult = code == 200 ? "OK — backend reachable" : "HTTP \(code)"
        } catch {
            testResult = "Error: \(error.localizedDescription)"
        }
    }
}
