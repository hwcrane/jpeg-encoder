use ndarray::{Array2, ArrayView2};

/// Applies a level shift for all emements in an array.
/// Subtracts 128 from each element so that they are bounded -127:127 rather than 0:255
pub fn level_shift(layer: ArrayView2<u8>) -> Array2<i8> {
    layer.map(|n| (*n as i16 - 128) as i8)
}

/// Applies 4:2:0 chroma subsampling
pub fn chroma_subsample(layer: ArrayView2<u8>) -> Array2<u8> {
    let shape = layer.shape();
    let mut result: Array2<u8> = Array2::zeros([(shape[0] + 1) / 2, (shape[1] + 1) / 2]);

    layer.indexed_iter().for_each(|((x, y), value)| {
        // Check if element is in the top left corner of the subgrid
        if x % 2 == 0 && y % 2 == 0 {
            result[(x / 2, y / 2)] = *value;
        }
    });

    result
}

/// Converts from RGB to YcBcR colourspace
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let y = (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8;
    let cb = (-0.16874 * r as f64 - 0.33126 * g as f64 + 0.5 * b as f64 + 128.) as u8;
    let cr = (0.5 * r as f64 - 0.41869 * g as f64 - 0.08131 * b as f64 + 128.) as u8;

    (y, cb, cr)
}

#[cfg(test)]
mod tests {
    use crate::colour::*;
    use ndarray::{array, Array2};

    #[test]
    /// Tests `level_shift`
    fn level_shift_test() {
        let input = array![[128, 255, 0, 54], [220, 101, 200, 90]];
        let expected_output = array![[0, 127, -128, -74], [92, -27, 72, -38]];

        assert_eq!(level_shift(input.view()), expected_output)
    }

    #[test]
    /// Tests `chroma_subsample` using input with dimensions divisible by 2
    fn chroma_subsample_test_1() {
        let input: Array2<u8> = array![
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ];
        let expected_output: Array2<u8> = array![[10, 30], [90, 110]];

        assert_eq!(chroma_subsample(input.view()), expected_output);
    }

    #[test]
    /// Tests `chroma_subsample` using input with dimensions not divisible by 2
    fn chroma_subsample_test_2() {
        let input: Array2<u8> = array![
            [10, 20, 30, 40, 50],
            [60, 70, 80, 90, 100],
            [110, 120, 130, 140, 150],
        ];
        let expected_output: Array2<u8> = array![[10, 30, 50], [110, 130, 150]];

        assert_eq!(chroma_subsample(input.view()), expected_output);
    }

    #[test]
    /// Tests `chroma_subsample` using empty input
    fn chroma_subsample_test_3() {
        let input: Array2<u8> = Array2::zeros((0, 0));
        let expected_output: Array2<u8> = Array2::zeros((0, 0));
        assert_eq!(chroma_subsample(input.view()), expected_output);
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!((y, cb, cr), (0, 128, 128));
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!((y, cb, cr), (255, 128, 128));
    }

    #[test]
    fn test_rgb_to_ycbcr_red() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
        assert_eq!((y, cb, cr), (76, 84, 255));
    }

    #[test]
    fn test_rgb_to_ycbcr_green() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 255, 0);
        assert_eq!((y, cb, cr), (149, 43, 21));
    }

    #[test]
    fn test_rgb_to_ycbcr_blue() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 255);
        assert_eq!((y, cb, cr), (29, 255, 107));
    }
}
