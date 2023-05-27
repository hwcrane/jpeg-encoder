use image::open;
use ndarray::Array2;

use crate::colour::{chroma_subsample, level_shift, rgb_to_ycbcr};

pub struct YcbcrImage {
    pub y: Array2<i8>,
    pub cb: Array2<i8>,
    pub cr: Array2<i8>,
}

impl YcbcrImage {
    pub fn open(path: &str) -> YcbcrImage {
        let rgb_image = open(path).unwrap().to_rgb8();

        let width = rgb_image.width() as usize;
        let height = rgb_image.height() as usize;

        let mut y_mat = Array2::<u8>::zeros([height, width]);
        let mut cb_mat = Array2::<u8>::zeros([height, width]);
        let mut cr_mat = Array2::<u8>::zeros([height, width]);

        for (j, i, pixel) in rgb_image.enumerate_pixels() {
            let (y, cb, cr) = rgb_to_ycbcr(pixel[0], pixel[1], pixel[2]);

            y_mat[(i as usize, j as usize)] = y;
            cb_mat[(i as usize, j as usize)] = cb;
            cr_mat[(i as usize, j as usize)] = cr;
        }

        cb_mat = chroma_subsample(cb_mat.view());
        cr_mat = chroma_subsample(cr_mat.view());

        YcbcrImage {
            y: level_shift(y_mat.view()),
            cb: level_shift(cb_mat.view()),
            cr: level_shift(cr_mat.view()),
        }
    }
}
