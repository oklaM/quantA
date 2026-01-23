use built::*;

fn main() {
    // 生成构建信息
    built::write_built_file().expect("Failed to acquire build-time information");
}
