use clap::{Parser, Subcommand};
use find::Config;
use win::get_system_times;
mod find;
mod win;

#[derive(Parser)]
#[command(
    version = "0.0.1",
    about = "Toolpack is a collection of tools for Rust developers"
)]
struct Cli {
    #[arg(short, long, default_value = "0")]
    verbose: u8,
}

#[test]
fn test_clap() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();
}
