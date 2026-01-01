#![allow(clippy::manual_retain)]

use clap::Parser;
use cli::Cli;
use onnxbench::{benchmark, setup_log};
use tracing::{error, warn};
mod cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    setup_log(&cli.log);

    let input_shape = cli.parse_input_shape();

    match input_shape {
        Ok(shape) => {
            match benchmark(&cli.model_path, cli.loop_n, shape, &cli.device).await {
                Ok(_) => {}
                Err(e) => warn!("{:}", e),
            };
        }
        Err(e) => {
            error!("{:}", e);
        }
    };
}
