export default function ClusterPage() {
  return (
    <div className="dashboard-content">
      <header className="content-header">
        <div>
          <h1 className="content-title">Cluster Orchestration</h1>
          <p className="content-description">
            V2 Hardware Cluster: Command and map multi-node GPU targets across local Mac, Linux, and Cloud environments.
          </p>
        </div>
      </header>
      <div className="panel" style={{ padding: "4rem", textAlign: "center" }}>
        <p style={{ opacity: 0.6 }}>Distributed controls coming in V2.1</p>
      </div>
    </div>
  );
}
