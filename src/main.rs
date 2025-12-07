#![allow(clippy::manual_retain)]

use clap::Parser;
use cli::Cli;
use onnxbench::{benchmark, setup_log};
use tracing::error;
mod cli;

fn main() {
    let cli = Cli::parse();

    setup_log(&cli.log);

    let input_shape = cli.parse_input_shape();

    match input_shape {
        Ok(shape) => {
            let _res = benchmark(&cli.model_path, cli.loop_n, shape);
        }
        Err(e) => {
            error!("{:}", e);
        }
    };
}
