# onnxbench

```bash
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

upx --best --lzma target/release/onnxbench # 20MB -> 5MB
```
