use indicatif::{ProgressBar, ProgressStyle};
use ndarray::ArrayD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::borrow::Cow;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};
use time::{format_description, UtcOffset};
use tracing::{debug, info};
use tracing_subscriber::fmt::time::OffsetTime;

pub mod cli;

pub fn benchmark(
    model_path: &String,
    loop_num: usize,
    input_shape: HashMap<String, Vec<usize>>,
) -> Result<(), usize> {
    // #[cfg(feature = "backend-candle")]
    // ort::set_api(ort_candle::api());
    // #[cfg(feature = "backend-tract")]
    // ort::set_api(ort_tract::api());

    // #[cfg(all(not(feature = "backend-candle"), not(feature = "backend-tract")))]
    ort::init()
        .with_execution_providers([
            // #[cfg(feature = "tensorrt")]
            // TensorRTExecutionProvider::default().build(),
            // #[cfg(feature = "cuda")]
            // CUDAExecutionProvider::default().build(),
            // #[cfg(feature = "onednn")]
            // OneDNNExecutionProvider::default().build(),
            // #[cfg(feature = "acl")]
            // ACLExecutionProvider::default().build(),
            // #[cfg(feature = "openvino")]
            // OpenVINOExecutionProvider::default().build(),
            // #[cfg(feature = "coreml")]
            // CoreMLExecutionProvider::default().build(),
            // #[cfg(feature = "rocm")]
            // ROCmExecutionProvider::default().build(),
            // #[cfg(feature = "cann")]
            // CANNExecutionProvider::default().build(),
            // #[cfg(feature = "directml")]
            // DirectMLExecutionProvider::default().build(),
            // #[cfg(feature = "tvm")]
            // TVMExecutionProvider::default().build(),
            // #[cfg(feature = "nnapi")]
            // NNAPIExecutionProvider::default().build(),
            // #[cfg(feature = "qnn")]
            // QNNExecutionProvider::default().build(),
            // #[cfg(feature = "xnnpack")]
            // XNNPACKExecutionProvider::default().build(),
            // #[cfg(feature = "armnn")]
            // ArmNNExecutionProvider::default().build(),
            // #[cfg(feature = "migraphx")]
            // MIGraphXExecutionProvider::default().build(),
            // #[cfg(feature = "vitis")]
            // VitisAIExecutionProvider::default().build(),
            // #[cfg(feature = "rknpu")]
            // RKNPUExecutionProvider::default().build(),
            // #[cfg(feature = "webgpu")]
            // WebGPUExecutionProvider::default().build(),
        ])
        .commit()
        .unwrap();

    let mut model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();

    info!("User input shape: {:?}", input_shape);

    for inputi in &model.inputs {
        info!(
            "Model input name: {:?}, size: {:?}, dtype: {:?}",
            inputi.name,
            inputi.input_type.tensor_shape().unwrap(),
            inputi.input_type.tensor_type().unwrap()
        );
    }

    // inference
    let mut costs: Vec<Duration> = Vec::new();

    let bar = ProgressBar::new(loop_num as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("Steps {pos:>}/{len}: {bar:50} [{elapsed_precise}/{eta_precise}]")
            .unwrap(),
    );
    info!("Start benchmark...");
    for _ in 0..loop_num {
        let empty_input: HashMap<&str, ArrayD<f32>> = input_shape
            .iter()
            .filter(|(_, shape)| !shape.is_empty())
            .map(|(name, shape)| (name.as_str(), ArrayD::<f32>::zeros(ndarray::IxDyn(&shape))))
            .collect();

        let mut model_input: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![];
        for (key, value) in empty_input {
            model_input.append(&mut ort::inputs![key=>Tensor::from_array(value).unwrap()]);
        }

        let start = Instant::now();
        let _outputs = model.run(model_input).unwrap();
        let elapsed = start.elapsed();

        costs.push(elapsed);
        bar.inc(1);
    }
    bar.finish();
    info!("End benchmark");

    costs.sort();
    let mean = round(
        costs.iter().sum::<Duration>().as_secs_f64() / loop_num as f64 * 1_000.0,
        6,
    );
    let perf = vec![
        PerformanceSummary {
            label: "Cost time(ms)",
            mean: mean,
            min: round(costs.iter().min().unwrap().as_secs_f64() * 1_000.0, 6),
            max: round(costs.iter().max().unwrap().as_secs_f64() * 1_000.0, 6),
            p90: round(
                costs[((costs.len() as f64 * 0.9) - 1.0).max(0.) as usize].as_secs_f64() * 1_000.0,
                6,
            ),
        },
        PerformanceSummary {
            label: "Throughput(tps)",
            mean: round(1_000.0 * 1.0 / mean, 6),
            min: -0f64,
            max: -0f64,
            p90: -0f64,
        },
    ];
    let table = Table::new(perf);
    println!("{:}", table);
    info!("Success benchmark");
    Ok(())
}

fn round(num: f64, round: u32) -> f64 {
    (num * 10_i32.pow(round) as f64).round() / 10_i32.pow(round) as f64
}

/// ms
#[derive(Tabled)]
struct PerformanceSummary {
    label: &'static str,
    mean: f64,
    min: f64,
    max: f64,
    p90: f64,
}

pub fn setup_log(level: &String) {
    match tracing_subscriber::fmt()
        .with_max_level(if level == "debug" {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .with_timer(OffsetTime::new(
            UtcOffset::current_local_offset().unwrap(),
            format_description::parse(
                "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]",
            )
            .unwrap(),
        ))
        .with_ansi(true)
        .with_env_filter("onnxbench=info")
        // .with_target(false)
        .with_line_number(false)
        .try_init()
    {
        Ok(_) => {
            debug!("Init tracing_subscriber successful!")
        }
        Err(_) => {
            debug!("Init tracing_subscriber failed!")
        }
    }
}
