import SwiftUI

struct MenuBarStatusView: View {
    @ObservedObject var store: NativeWorkspaceStore
    let bridge: DesktopBridge
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            statsGrid
            Divider()
            quickLinks
            Divider()
            footer
        }
        .frame(width: 280)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(store.apiReachable ? .green : .red)
            VStack(alignment: .leading, spacing: 2) {
                Text("AI-Factory")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                Text(store.apiReachable ? "Backend online · v\(store.apiVersion)" : "Backend unreachable")
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Circle()
                .fill(store.apiReachable ? Color.green : Color.red)
                .frame(width: 8, height: 8)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var statsGrid: some View {
        HStack(spacing: 0) {
            statCell(value: "\(store.instanceCount)", label: "Instances", icon: "square.stack.3d.up")
            Divider().frame(height: 36)
            statCell(value: "\(store.runningCount)", label: "Running", icon: "bolt.fill")
            Divider().frame(height: 36)
            statCell(value: store.apiReachable ? store.formattedUptime : "—", label: "Uptime", icon: "clock")
        }
        .padding(.vertical, 8)
    }

    private func statCell(value: String, label: String, icon: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
            Text(label)
                .font(.system(size: 10, weight: .regular, design: .rounded))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private var quickLinks: some View {
        VStack(alignment: .leading, spacing: 2) {
            menuRow(icon: "globe", label: "Open workspace") {
                bridge.open(store.dashboardURL)
            }
            menuRow(icon: "server.rack", label: "Open API root") {
                bridge.open(store.apiURL)
            }
            menuRow(icon: "folder", label: "Reveal artifacts") {
                bridge.reveal(store.artifactsURL)
            }
            menuRow(icon: "terminal", label: "Launch TUI") {
                bridge.runShellCommand("python -m ai_factory.tui")
            }
            menuRow(icon: "arrow.clockwise", label: "Refresh status") {
                Task { await store.fetchStatus() }
            }
        }
        .padding(.vertical, 4)
    }

    private func menuRow(icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .frame(width: 16)
                    .foregroundStyle(.blue)
                Text(label)
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 14)
        .padding(.vertical, 5)
        .background(Color.primary.opacity(0.001))
    }

    private var footer: some View {
        HStack {
            if let checked = store.lastChecked {
                Text("Updated \(checked, style: .relative) ago")
                    .font(.system(size: 10, weight: .regular, design: .rounded))
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Button("Quit") { NSApplication.shared.terminate(nil) }
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}
