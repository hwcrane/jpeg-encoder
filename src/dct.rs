use std::{f64::consts::PI, ops::Div};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

pub fn quantise(dtc: ArrayView2<f64>, quantisation_matrix: ArrayView2<f64>) -> Array2<i32> {
    (&dtc / &quantisation_matrix).map(|&n| n as i32)
}

pub fn dct_2d(cell: ArrayView2<f64>) -> Array2<f64> {
    let mut rows = Array2::<f64>::from_elem((8, 8), 0.);
    let mut cols = Array2::<f64>::from_elem((8, 8), 0.);

    for row in 0..8 {
        rows.slice_mut(s![row, ..]).assign(&dct_1d(cell.row(row)));
    }

    for col in 0..8 {
        cols.slice_mut(s![.., col])
            .assign(&dct_1d(rows.column(col)));
    }

    cols
}

fn dct_1d(input: ArrayView1<f64>) -> Array1<f64> {
    let n = input.len();
    let sqrt2_over_n = (2.0 / n as f64).sqrt();

    (0..n)
        .map(|k| {
            let cu = if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 };
            let sum = input.iter().enumerate().fold(0.0, |total, (n, &xn)| {
                total + xn * (((2 * n + 1) as f64 * PI * k as f64) / 16.).cos()
            });
            sqrt2_over_n * cu * sum
        })

        .collect::<Array1<f64>>()
}
