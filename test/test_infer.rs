use std::path::Path;

use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis};
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, AsImageView, WindowOptions};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod common;

const YOLOV8M_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/yolov8m.onnx";


#[cfg(test)]
mod yolo {
    use super::*;

    #[test]
    fn test_infer() {
        
    }
}
