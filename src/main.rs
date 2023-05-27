use dct_image_compression::encode;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let filepath = &args[1];

    encode(filepath)
}
