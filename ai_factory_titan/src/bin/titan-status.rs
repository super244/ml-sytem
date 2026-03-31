use ai_factory_titan::{detect_hardware, TitanScheduler};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let hardware = detect_hardware();
    let scheduler = TitanScheduler::status();
    let payload = serde_json::json!({
        "hardware": hardware,
        "scheduler": scheduler,
        "neural_accelerator": {
            "target": "apple-amx",
            "mode": "prompt-preprocess",
            "speedup_goal_vs_m4": 4
        },
        "quantization": {
            "formats": ["4bit", "8bit"],
            "layout": "arrow-columnar"
        }
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
