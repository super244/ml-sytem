import Charts
import SwiftUI

private let desktopMetricsDemoModeEnabled =
    ProcessInfo.processInfo.environment["AI_FACTORY_DESKTOP_DEMO_MODE"] == "1" ||
    ProcessInfo.processInfo.environment["AI_FACTORY_DEMO_MODE"] == "1"

struct MetricSample: Identifiable {
    let id = UUID()
    let timestamp: Date
    let value: Double
}

struct FlexibleMetricValue: Decodable {
    let value: Double

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
            return
        }
        if let intValue = try? container.decode(Int.self) {
            value = Double(intValue)
            return
        }
        throw DecodingError.typeMismatch(
            Double.self,
            DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected numeric metric value")
        )
    }
}

struct MetricSeries {
    var label: String
    var unit: String
    var color: Color
    var samples: [MetricSample] = []
    var maxCapacity: Int = 60

    var latest: Double { samples.last?.value ?? 0 }
    var peak: Double { samples.map(\.value).max() ?? 0 }
    var average: Double {
        guard !samples.isEmpty else { return 0 }
        return samples.map(\.value).reduce(0, +) / Double(samples.count)
    }

    mutating func append(_ value: Double) {
        samples.append(MetricSample(timestamp: Date(), value: value))
        if samples.count > maxCapacity { samples.removeFirst() }
    }
}

@MainActor
final class MetricsStore: ObservableObject {
    @Published var accuracy = MetricSeries(label: "Accuracy", unit: "%", color: .blue)
    @Published var latency = MetricSeries(label: "Latency", unit: "s", color: .purple)
    @Published var throughput = MetricSeries(label: "Tokens/s", unit: "tok/s", color: .green)
    @Published var activity = MetricSeries(label: "Activity", unit: "count", color: .red)
    @Published var isLive = false
    @Published var fetchError: String? = nil
    @Published var targetInstanceName: String = "No instance selected"

    private var timer: Timer?
    private let apiURL: URL
    private let demoMode = desktopMetricsDemoModeEnabled

    init(apiURL: URL) {
        self.apiURL = apiURL
    }

    func startLive() {
        isLive = true
        tick()
        timer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    func stopLive() {
        isLive = false
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        Task { await fetchMetrics() }
    }

    func fetchMetrics() async {
        do {
            let instanceID = try await resolveTargetInstanceID()
            let endpoint = apiURL.appendingPathComponent("v1/instances/\(instanceID)/metrics")
            var req = URLRequest(url: endpoint)
            req.timeoutInterval = 4
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                fetchError = "HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)"
                if demoMode { seedSimulated() }
                return
            }
            let decoded = try JSONDecoder().decode(InstanceMetricsResponse.self, from: data)
            let summary = decoded.numericSummary
            if let accuracyValue = summary["accuracy"] ?? summary["parse_rate"] {
                accuracy.append(accuracyValue * 100)
            }
            if let latencyValue = summary["avg_latency_s"] {
                latency.append(latencyValue)
            }
            if let throughputValue = summary["avg_tokens_per_second"] {
                throughput.append(throughputValue)
            }
            if let activityValue = summary["requests"] ?? summary["latest_step"] ?? summary["records"] {
                activity.append(activityValue)
            }
            fetchError = nil
        } catch {
            fetchError = error.localizedDescription
            if demoMode { seedSimulated() }
        }
    }

    private func seedSimulated() {
        accuracy.append(Double.random(in: 40...95))
        latency.append(Double.random(in: 0.1...3.0))
        throughput.append(Double.random(in: 1...80))
        activity.append(Double.random(in: 1...500))
    }

    private func resolveTargetInstanceID() async throws -> String {
        var request = URLRequest(url: apiURL.appendingPathComponent("v1/instances"))
        request.timeoutInterval = 4
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        let decoded = try JSONDecoder().decode(APIInstancesResponse.self, from: data)
        guard let target = decoded.instances.first(where: { $0.status == "running" }) ?? decoded.instances.first else {
            throw NSError(domain: "AIFactoryNative", code: 404, userInfo: [NSLocalizedDescriptionKey: "No instances are available for metrics."])
        }
        targetInstanceName = target.name
        return target.id
    }
}

struct InstanceMetricsResponse: Decodable {
    let summary: [String: FlexibleMetricValue]

    var numericSummary: [String: Double] {
        Dictionary(uniqueKeysWithValues: summary.map { ($0.key, $0.value.value) })
    }
}

struct MetricsChartView: View {
    @StateObject private var metrics: MetricsStore
    @State private var selectedSeries: String = "Accuracy"

    init(apiURL: URL) {
        _metrics = StateObject(wrappedValue: MetricsStore(apiURL: apiURL))
    }

    private var allSeries: [MetricSeries] {
        [metrics.accuracy, metrics.latency, metrics.throughput, metrics.activity]
    }

    private var activeSeries: MetricSeries {
        allSeries.first { $0.label == selectedSeries } ?? metrics.accuracy
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            headerRow
            seriesPicker
            sparklineCard(series: activeSeries)
            summaryGrid
        }
        .padding(24)
        .onAppear { metrics.startLive() }
        .onDisappear { metrics.stopLive() }
    }

    private var headerRow: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Live Metrics")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("\(metrics.targetInstanceName) · sampled every 3 seconds · last \(metrics.accuracy.samples.count) points")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 6) {
                Circle()
                    .fill(metrics.isLive ? Color.green : Color.secondary)
                    .frame(width: 8, height: 8)
                Text(metrics.isLive ? "Live" : "Paused")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Button(metrics.isLive ? "Pause" : "Resume") {
                metrics.isLive ? metrics.stopLive() : metrics.startLive()
            }
            .buttonStyle(.bordered)
        }
    }

    private var seriesPicker: some View {
        HStack(spacing: 10) {
            ForEach(allSeries, id: \.label) { s in
                Button {
                    selectedSeries = s.label
                } label: {
                    HStack(spacing: 6) {
                        Circle().fill(s.color).frame(width: 8, height: 8)
                        Text(s.label)
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        selectedSeries == s.label
                            ? s.color.opacity(0.15)
                            : Color.primary.opacity(0.05),
                        in: Capsule()
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func sparklineCard(series: MetricSeries) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(series.label)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                Text("(\(series.unit))")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f %@", series.latest, series.unit))
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(series.color)
            }
            if series.samples.isEmpty {
                Text("Waiting for data…")
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 140, alignment: .center)
            } else {
                Chart(series.samples) { sample in
                    AreaMark(
                        x: .value("Time", sample.timestamp),
                        y: .value(series.label, sample.value)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [series.color.opacity(0.4), series.color.opacity(0.05)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    LineMark(
                        x: .value("Time", sample.timestamp),
                        y: .value(series.label, sample.value)
                    )
                    .foregroundStyle(series.color)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { v in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [4]))
                            .foregroundStyle(Color.primary.opacity(0.1))
                        AxisValueLabel {
                            if let d = v.as(Double.self) {
                                Text(String(format: "%.0f", d))
                                    .font(.system(size: 10, design: .rounded))
                            }
                        }
                    }
                }
                .frame(height: 140)
            }
        }
        .padding(18)
        .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.black.opacity(0.06))
        )
    }

    private var summaryGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 14)], spacing: 14) {
            ForEach(allSeries, id: \.label) { s in
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Circle().fill(s.color).frame(width: 8, height: 8)
                        Text(s.label)
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                    }
                    Text(String(format: "%.1f %@", s.latest, s.unit))
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(s.color)
                    HStack {
                        miniStat(label: "avg", value: String(format: "%.1f", s.average))
                        miniStat(label: "peak", value: String(format: "%.1f", s.peak))
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.white.opacity(0.72), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(s.color.opacity(0.2))
                )
            }
        }
    }

    private func miniStat(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label.uppercased())
                .font(.system(size: 9, weight: .semibold, design: .rounded))
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
