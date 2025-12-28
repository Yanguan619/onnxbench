# onnxbench

## Build

```bash
cargo build --release
upx --best --lzma target/release/onnxbench # 20MB -> 5MB
```

### (Option) Cross build

```bash
# x64
cargo build --release --target x86_64-unknown-linux-gnu
# aarch64
cargo build --release --target aarch64-unknown-linux-gnu

upx --best --lzma target/release/onnxbench # 20MB -> 5MB
```

## Usage

```bash

onnxbench --model-path .assets/yolov5nu.onnx --input-shape "images:16,3,256,256"
```

### Help

```bash
onnxbench --help
```
