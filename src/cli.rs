use std::collections::HashMap;

use clap::Parser;

#[derive(Parser)]
#[command(
    version = "0.0.1",
    about = "onnxbench is a collection of tools for Rust developers"
)]
pub struct Cli {
    #[arg(short, long, help = "e.g: .assets/yolov5nu.onnx")]
    pub model_path: String,
    #[arg(short, long, default_value = "", help = "e.g: images:1,3,640,640")]
    pub input_shape: String,
    #[arg(long, default_value = "cpu")]
    pub device: String,
    #[arg(short, long = "loop", default_value = "10", help = "e.g: 100")]
    pub loop_n: usize,
    #[arg(long, default_value = "info")]
    pub log: String,
}

impl Cli {
    pub fn parse_input_shape(&self) -> Result<HashMap<String, Vec<usize>>, &str> {
        let mut input_shape: HashMap<String, Vec<usize>> = HashMap::new();
        let x = self.input_shape.split(";").collect::<Vec<&str>>();
        for i in 0..x.len() {
            let temp: Vec<&str> = x[i].split(":").collect();

            if self.input_shape.is_empty() {
                return Ok(input_shape);
            }

            if !x[i].contains(":") || temp[1].len() == 0 {
                return Err("Parse `input-shape` arg failed!");
            }

            input_shape.insert(
                temp[0].to_string(),
                temp[1].split(",").map(|x| x.parse().unwrap()).collect(),
            );
        }
        Ok(input_shape)
    }
}
