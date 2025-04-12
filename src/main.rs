#![allow(clippy::manual_retain)]

use clap::{Parser, Subcommand};
use image::{imageops::FilterType, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array, Axis, Dim, Dimension, IntoDimension};
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, AsImageView, WindowOptions};
use std::{collections::HashMap, iter::Map, path::Path, time::Duration};
use std::{process::exit, time::Instant};
use tracing::{debug, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
mod common;

#[derive(Parser)]
#[command(
    version = "0.0.1",
    about = "Toolpack is a collection of tools for Rust developers"
)]
struct Cli {
    #[arg(short, long, default_value = "0")]
    verbose: u8,
    #[arg(
        short,
        long,
        default_value = "D:/mCloudDownload/dl/weights/yolo/yolov5nu.onnx"
    )]
    model_path: String,
    #[arg(short, long, default_value = "100")]
    loop_num: usize,
    #[arg(short, long, default_value = "images:1,3,640,640")]
    input_shape: String,
    #[arg(long, default_value = "info")]
    level: String,
}

fn benchmark(model_path: &str, loop_num: usize, input_shape: HashMap<String, Vec<usize>>) {
    common::init().unwrap();

    let model = Session::builder()
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();

    let mut exit_flag = true;
    for inputi in &model.inputs {
        debug!("input name : {:?}", inputi.name);
        debug!(
            "input: {:?}, dtype: {:?}",
            inputi.input_type.tensor_dimensions().unwrap(),
            inputi.input_type.tensor_type().unwrap()
        );
    }

    // inference
    let mut costs: Vec<Duration> = Vec::new();

    let bar = ProgressBar::new(loop_num as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("Infer: {bar:60} {pos:>}/{len} [{elapsed_precise}/{eta_precise}]")
            .unwrap(),
    );
    for _ in 0..loop_num {
        let input = ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

        let start = Instant::now();
        let outputs = model
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
    info!(
        "first elapsed: {:.2?}, average elapsed: {:.2?}, last elapsed: {:.2?}",
        cost_first, cost_average, cost_last
    )
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

    let mut input_shape: HashMap<String, Vec<usize>> = HashMap::new();
    let x = cli.input_shape.split(";").collect::<Vec<&str>>();
    for i in 0..x.len() {
        let temp: Vec<&str> = x[i].split(":").collect();
        input_shape.insert(
            temp[0].to_string(),
            temp[1].split(",").map(|x| x.parse().unwrap()).collect(),
        );
    }
    debug!("{:?}", input_shape);

    benchmark(cli.model_path.as_str(), cli.loop_num, input_shape);
}

// #[cfg(test)]
// fn test_benchmark() {
//     const YOLOV8M_URL: &str = "D:/mCloudDownload/dl/weights/yolo/yolov5nu.onnx";
//     benchmark(YOLOV8M_URL);
// }
