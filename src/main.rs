#![allow(clippy::manual_retain)]

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use ort::session::Session;
use std::time::Instant;
use std::{collections::HashMap, time::Duration};
use tracing::debug;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
mod common;

#[derive(Parser)]
#[command(
    version = "0.0.1",
    about = "onnxbench is a collection of tools for Rust developers"
)]
struct Cli {
    #[arg(short, long, default_value = ".assets/yolov5nu.onnx")]
    model_path: String,
    #[arg(short, long, default_value = "100")]
    loop_num: usize,
    #[arg(short, long, default_value = "images:1,3,640,640")]
    input_shape: String,
    #[arg(long, default_value = "info")]
    level: String,
}

impl Cli {
    fn parse_input_shape(&self) -> HashMap<String, Vec<usize>> {
        let mut input_shape: HashMap<String, Vec<usize>> = HashMap::new();
        let x = self.input_shape.split(";").collect::<Vec<&str>>();
        for i in 0..x.len() {
            let temp: Vec<&str> = x[i].split(":").collect();
            input_shape.insert(
                temp[0].to_string(),
                temp[1].split(",").map(|x| x.parse().unwrap()).collect(),
            );
        }
        input_shape
    }
}

fn benchmark(
    model_path: &str,
    loop_num: usize,
    input_shape: HashMap<String, Vec<usize>>,
) -> Result<(), usize> {
    common::init().unwrap();

    let model = Session::builder()
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();

    for inputi in &model.inputs {
        debug!("input name : {:?}", inputi.name);
        debug!(
            "input: {:?}, dtype: {:?}",
            inputi.input_type.tensor_dimensions().unwrap(),
            inputi.input_type.tensor_type().unwrap()
        );
    }

    debug!("user input shape: {:?}", input_shape);

    // inference
    let mut costs: Vec<Duration> = Vec::new();

    let bar = ProgressBar::new(loop_num as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("Steps {pos:>}/{len}: {bar:50} [{elapsed_precise}/{eta_precise}]")
            .unwrap(),
    );
    println!("Start benchmark...");
    for _ in 0..loop_num {
        let input = ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

        let start = Instant::now();
        let _outputs = model
            .run(
                ort::inputs! {
                    "images" => input
                }
                .unwrap(),
            )
            .unwrap();

        let elapsed = start.elapsed();

        costs.push(elapsed);
        bar.inc(1);
    }
    bar.finish();
    let cost_first = costs[0];
    let cost_last = costs.iter().last().unwrap();
    let cost_average = costs.iter().sum::<Duration>() / loop_num as u32;
    println!(
        "End benchmark, first elapsed: {:.2?}, average elapsed: {:.2?}, last elapsed: {:.2?}.",
        cost_first, cost_average, cost_last
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    // Initialize tracing to receive debug messages from `ort`
    let x = format!(
        "{},ort={}",
        cli.level.to_lowercase(),
        cli.level.to_lowercase()
    );
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| x.into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let input_shape = cli.parse_input_shape();
    debug!("{:?}", input_shape);

    let _res = benchmark(cli.model_path.as_str(), cli.loop_num, input_shape);
}
#[test]
fn test_benchmark() {
    let inn = Cli {
        model_path: ".assets/yolov5nu.onnx".to_string(),
        loop_num: 2,
        level: "debug".to_string(),
        input_shape: "images:1,3,640,640".to_string(),
    };
    let res = benchmark(
        inn.model_path.as_str(),
        inn.loop_num,
        inn.parse_input_shape(),
    );
    assert_eq!(res, Ok(()));
}
