import SwiftUI

struct MainTabView: View {
    @ObservedObject var store: NativeWorkspaceStore
    let bridge: DesktopBridge
    @State private var selectedTab: AppTab = .dashboard

    enum AppTab: String, CaseIterable, Identifiable {
        case dashboard = "Dashboard"
        case instances = "Instances"
        case metrics   = "Metrics"
        case logs      = "Logs"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .dashboard:  "house.fill"
            case .instances:  "square.stack.3d.up.fill"
            case .metrics:    "chart.xyaxis.line"
            case .logs:       "text.alignleft"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detailView
        }
        .frame(minWidth: 1060, minHeight: 720)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            sidebarHeader
            Divider().padding(.vertical, 8)
            ForEach(AppTab.allCases) { tab in
                sidebarRow(tab)
            }
            Spacer()
            sidebarFooter
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 16)
        .frame(width: 200)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var sidebarHeader: some View {
        HStack(spacing: 10) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 1) {
                Text("AI-Factory")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                Text(store.apiReachable ? "Online" : "Offline")
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(store.apiReachable ? .green : .red)
            }
        }
        .padding(.horizontal, 4)
        .padding(.bottom, 4)
    }

    private func sidebarRow(_ tab: AppTab) -> some View {
        Button {
            selectedTab = tab
        } label: {
            HStack(spacing: 10) {
                Image(systemName: tab.icon)
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: 20)
                    .foregroundStyle(selectedTab == tab ? .blue : .secondary)
                Text(tab.rawValue)
                    .font(.system(size: 13, weight: selectedTab == tab ? .semibold : .regular, design: .rounded))
                    .foregroundStyle(selectedTab == tab ? .primary : .secondary)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                selectedTab == tab
                    ? Color.blue.opacity(0.1)
                    : Color.clear,
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var sidebarFooter: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            HStack(spacing: 6) {
                Circle()
                    .fill(store.apiReachable ? Color.green : Color.red)
                    .frame(width: 7, height: 7)
                Text(store.apiReachable ? "v\(store.apiVersion)" : "No backend")
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    Task { await store.fetchStatus() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Refresh status")
            }
            .padding(.horizontal, 4)
        }
    }

    @ViewBuilder
    private var detailView: some View {
        switch selectedTab {
        case .dashboard:
            ControlCenterView(store: store, bridge: bridge)
        case .instances:
            InstanceListView(apiURL: store.apiURL)
        case .metrics:
            MetricsChartView(apiURL: store.apiURL)
        case .logs:
            LogStreamView(apiURL: store.apiURL)
        }
    }
}
