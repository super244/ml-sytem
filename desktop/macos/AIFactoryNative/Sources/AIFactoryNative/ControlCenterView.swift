import SwiftUI

struct ControlCenterView: View {
    @ObservedObject var store: NativeWorkspaceStore
    let bridge: DesktopBridge
    @State private var copiedLaunchCommand = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                statusBar
                actionGrid
                launchStrip
            }
            .padding(24)
        }
        .frame(minWidth: 980, minHeight: 720)
        .background(
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(red: 0.96, green: 0.98, blue: 0.97),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 18) {
            VStack(alignment: .leading, spacing: 10) {
                Text("AI-Factory")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Text("Native macOS control center")
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                Text(store.shellSummary)
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 620, alignment: .leading)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 10) {
                Label("SwiftUI shell", systemImage: "macwindow")
                Label("Electron fallback remains available", systemImage: "window.viewfinder")
                HStack(spacing: 6) {
                    Circle()
                        .fill(store.apiReachable ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(store.apiReachable ? "Backend connected" : "Backend unreachable")
                }
                .font(.system(size: 13, weight: .medium, design: .rounded))
            }
            .font(.system(size: 13, weight: .medium, design: .rounded))
            .foregroundStyle(.secondary)
        }
    }

    private var statusBar: some View {
        HStack(spacing: 16) {
            statusPill(
                icon: "dot.radiowaves.left.and.right",
                label: store.apiReachable ? "API Online" : "API Offline",
                color: store.apiReachable ? .green : .red
            )
            statusPill(
                icon: "square.stack.3d.up",
                label: "\(store.instanceCount) instances",
                color: .blue
            )
            statusPill(
                icon: "bolt.fill",
                label: "\(store.runningCount) running",
                color: store.runningCount > 0 ? .orange : .secondary
            )
            if store.apiReachable {
                statusPill(
                    icon: "clock",
                    label: "Up \(store.formattedUptime)",
                    color: .secondary
                )
                statusPill(
                    icon: "tag",
                    label: store.apiVersion,
                    color: .secondary
                )
            }
            Spacer()
            if let error = store.statusError {
                Text(error)
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(.red.opacity(0.8))
                    .lineLimit(1)
            }
            Button {
                Task { await store.fetchStatus() }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 13, weight: .medium))
            }
            .buttonStyle(.bordered)
            .help("Refresh backend status")
        }
        .padding(14)
        .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.black.opacity(0.06))
        )
    }

    private func statusPill(icon: String, label: String, color: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .rounded))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(color.opacity(0.1), in: Capsule())
    }

    private var actionGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 14)], spacing: 14) {
            ForEach(store.quickActions) { action in
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 8) {
                        Image(systemName: action.icon)
                            .font(.system(size: 20, weight: .medium))
                            .foregroundStyle(.blue)
                        Text(action.title)
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                    }
                    Text(action.detail)
                        .font(.system(size: 13, weight: .regular, design: .rounded))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Button("Open") {
                            if let url = URL(string: action.command), action.command.hasPrefix("http") {
                                bridge.open(url)
                            } else if action.command.hasPrefix("python") || action.command.hasPrefix("swift") {
                                bridge.runShellCommand(action.command)
                            } else {
                                bridge.reveal(URL(fileURLWithPath: action.command))
                            }
                        }
                        .buttonStyle(.borderedProminent)

                        Button("Copy") {
                            bridge.copy(action.command)
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(Color.black.opacity(0.06))
                )
            }
        }
    }

    private var launchStrip: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Native launch command")
                .font(.system(size: 18, weight: .semibold, design: .rounded))
            Text(store.launchCommand)
                .font(.system(size: 12, weight: .regular, design: .monospaced))
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.86), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .foregroundStyle(.white)
            HStack {
                Button(copiedLaunchCommand ? "Copied" : "Copy launch command") {
                    bridge.copy(store.launchCommand)
                    copiedLaunchCommand = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
                        copiedLaunchCommand = false
                    }
                }
                .buttonStyle(.borderedProminent)

                Button("Open workspace") {
                    bridge.open(store.dashboardURL)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.black.opacity(0.06))
        )
    }
}
