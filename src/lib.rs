use indicatif::{ProgressBar, ProgressStyle};
use ndarray::ArrayD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};
use time::{format_description, UtcOffset};
use tokio::signal;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use tracing_subscriber::fmt::time::OffsetTime;
pub mod cli;

pub async fn benchmark(
    model_path: &String,
    loop_num: usize,
    mut input_shape: HashMap<String, Vec<usize>>,
    device: &String,
) -> Result<(), String> {
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

    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();

    let use_dafault_input = input_shape.is_empty();
    let mut default_bs = 8;
    let mut format = ArrayFormat::ND;

    info!("Device: {}", device);
    for input in &model.inputs {
        let tensor_shape = input.input_type.tensor_shape().unwrap();
        let mut vv = tensor_shape.to_vec();
        if vv.len() == 4 {
            format = ArrayFormat::BCHW;
        }
        info!(
            "Model input name: {:?}, size: {:?}, dtype: {:?}",
            input.name,
            tensor_shape,
            input.input_type.tensor_type().unwrap()
        );
        if use_dafault_input {
            if format == ArrayFormat::BCHW && vv.get(0) == Some(&-1) {
                vv[0] = default_bs;
            };
            input_shape.insert(
                input.name.clone(),
                vv.iter()
                    .map(|x| if *x < 0 { 256 } else { *x as usize })
                    .collect(),
            );
        }
    }

    for i in input_shape.values() {
        if i.len() == 4 {
            default_bs = i[0] as i64;
        }
    }

    if !use_dafault_input {
        info!("User input shape: {:?}", input_shape);
    } else {
        warn!(
            "User input shape is not provided, use default input shape: {:?}",
            input_shape
        );
    };
    let res = forward(model, input_shape, default_bs, loop_num).await;

    res
}

async fn forward(
    mut model: Session,
    input_shape: HashMap<String, Vec<usize>>,
    bs: i64,
    loop_num: usize,
) -> Result<(), String> {
    let costs = Arc::new(Mutex::new(Vec::<Duration>::new()));
    let bar = Arc::new(ProgressBar::new(loop_num as u64));
    bar.set_style(
        ProgressStyle::default_bar()
            .template("Steps {pos:>}/{len}: {bar:50} [{elapsed_precise}/{eta_precise}]")
            .unwrap(),
    );
    info!("Start benchmark...");
    // 异步执行循环
    let cancelled = Arc::new(AtomicBool::new(false));
    let costs_clone = Arc::clone(&costs);
    let bar_clone = Arc::clone(&bar);
    let cancelled_clone = Arc::clone(&cancelled);
    let handle = tokio::spawn(async move {
        let empty_input: HashMap<&str, ArrayD<f32>> = input_shape
            .iter()
            .filter(|(_, shape)| !shape.is_empty())
            .map(|(name, shape)| (name.as_str(), ArrayD::<f32>::zeros(ndarray::IxDyn(&shape))))
            .collect();
        for _ in 0..loop_num {
            // 检查是否取消
            if cancelled_clone.load(Ordering::Relaxed) {
                break;
            }

            let mut model_input: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![];
            for (key, value) in empty_input.clone() {
                model_input.append(&mut ort::inputs![key=>Tensor::from_array(value).unwrap()]);
            }
            // forward
            let start = Instant::now();
            model.run(model_input).unwrap();
            let elapsed = start.elapsed();

            costs_clone.lock().await.push(elapsed);
            bar_clone.inc(1);
        }
    });

    let wait_task = tokio::spawn(async move { handle.await });

    tokio::select! {
        result = wait_task => {
            result.unwrap().unwrap();
        }
        _ = signal::ctrl_c() => {
            print!("\n");
            info!("Checked Ctrl+C");
            cancelled.store(true, Ordering::Relaxed);
        }
    }
    let mut costs = costs.lock().await;
    info!("End benchmark");

    costs.sort();
    let num_finish = costs.len();
    if num_finish == 0 {
        return Err("No enough benchmark's data to generate summary.".to_string());
    }
    let mut danwei = 1.0;
    let mut danweis = "s";
    let mean = costs.iter().sum::<Duration>().as_secs_f32() / num_finish as f32;
    if mean < 0.1 {
        danwei = 1_000.0;
        danweis = "ms";
    };
    let mean = round(mean * danwei, 3);
    let min = round(costs.iter().min().unwrap().as_secs_f32() * danwei, 3);
    let max = round(costs.iter().max().unwrap().as_secs_f32() * danwei, 3);
    let p90 = round(costs.ana(0.90).as_secs_f32() * danwei, 3);
    let p95 = round(costs.ana(0.95).as_secs_f32() * danwei, 3);
    let p99 = round(costs.ana(0.99).as_secs_f32() * danwei, 3);
    let num_label = if num_finish == loop_num {
        format!("{}", loop_num)
    } else {
        format!("{}/{}", num_finish, loop_num)
    };

    let bs_f = bs as f32;
    let perf = vec![
        PerformanceSummary {
            label: Box::leak(format!("Cost time({})", danweis).into_boxed_str()),
            mean: mean,
            min: min,
            max: max,
            p90: p90,
            p95: p95,
            p99: p99,
            num: num_label.clone(),
        },
        PerformanceSummary {
            label: "Throughput(tps)",
            mean: bs_f * round(danwei / mean, 3),
            min: bs_f * round(danwei / max, 3),
            max: bs_f * round(danwei / min, 3),
            p90: bs_f * round(danwei / p90, 3),
            p95: bs_f * round(danwei / p95, 3),
            p99: bs_f * round(danwei / p99, 3),
            num: num_label,
        },
    ];
    let table = Table::new(perf);
    info!("Success benchmark, summary benchmark result.");
    println!("{:}", table);
    Ok(())
}

trait Round {
    fn round_to(self, decimals: u32) -> Self;
}

impl Round for f32 {
    fn round_to(self, decimals: u32) -> Self {
        let multiplier = 10_f32.powi(decimals as i32);
        (self * multiplier).round() / multiplier
    }
}

impl Round for f64 {
    fn round_to(self, decimals: u32) -> Self {
        let multiplier = 10_f64.powi(decimals as i32);
        (self * multiplier).round() / multiplier
    }
}

fn round<T: Round>(num: T, decimals: u32) -> T {
    num.round_to(decimals)
}

/// ms
#[derive(Tabled)]
struct PerformanceSummary {
    label: &'static str,
    mean: f32,
    min: f32,
    max: f32,
    p90: f32,
    p95: f32,
    p99: f32,
    num: String,
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

trait SummaryData {
    fn ana(&self, w: f32) -> Duration;
}

impl SummaryData for Vec<Duration> {
    fn ana(&self, w: f32) -> Duration {
        let idx = ((self.len() as f32 * w) - 1.0).max(0.) as usize;
        self[idx]
    }
}

#[derive(PartialEq)]
enum ArrayFormat {
    BCHW,
    ND,
}
