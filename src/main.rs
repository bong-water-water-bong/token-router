//! Token Router — binary entry point.
//!
//! Parses CLI arguments, loads config, and starts the server.

use clap::Parser;
use token_router::{handlers, config::Config};
use tracing::info;

/// Token Router CLI.
#[derive(Parser, Debug)]
#[command(name = "token-router", about = "Token-level router for multi-backend LLM inference")]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "router.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load config
    let config = if std::path::Path::new(&args.config).exists() {
        Config::from_file(&args.config)?
    } else {
        info!("No config file found, using defaults");
        let mut cfg = Config::default_config();
        cfg.backends.insert(
            "default".to_string(),
            token_router::config::BackendConfig {
                backend_type: token_router::config::BackendType::OpenAI,
                base_url: "http://127.0.0.1:13305/v1".to_string(),
                api_key: None,
                models: vec!["*".to_string()],
                speed_tok_s: Some(100.0),
                cost_per_token: 1.0,
            },
        );
        cfg
    };

    // Initialize tracing (non-fatal if already set)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(&config.server.log_level)
        .try_init();

    // Run server
    handlers::run_server(config).await
}
