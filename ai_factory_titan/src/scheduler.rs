use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub struct GpuTask {
    pub label: String,
    pub reply: oneshot::Sender<String>,
}

#[derive(Clone)]
pub struct TitanScheduler {
    sender: mpsc::Sender<GpuTask>,
}

#[derive(Debug, Serialize)]
pub struct SchedulerStatus {
    pub runtime: &'static str,
    pub queue_policy: &'static str,
    pub ui_budget_hz: u16,
    pub max_inflight_tasks: usize,
    pub priority_bands: u8,
}

impl TitanScheduler {
    pub fn new(capacity: usize) -> Self {
        let (sender, mut receiver) = mpsc::channel::<GpuTask>(capacity);
        tokio::spawn(async move {
            while let Some(task) = receiver.recv().await {
                let _ = task.reply.send(format!("completed: {}", task.label));
            }
        });
        Self { sender }
    }

    pub async fn submit(&self, label: impl Into<String>) -> anyhow::Result<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.sender
            .send(GpuTask {
                label: label.into(),
                reply: reply_tx,
            })
            .await?;
        Ok(reply_rx.await?)
    }

    pub fn status() -> SchedulerStatus {
        SchedulerStatus {
            runtime: "tokio",
            queue_policy: "bounded-priority",
            ui_budget_hz: 120,
            max_inflight_tasks: 64,
            priority_bands: 3,
        }
    }
}
