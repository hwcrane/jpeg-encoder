use ndarray::{s, Array1, Array2, ArrayView2};

use crate::{
    dct::{dct_2d, quantise},
    image_buffer::YcbcrImage,
    quantisation_tables,
};

pub fn encode(path: &str) {
    let ycbcr = YcbcrImage::open(path);

    let y_cells = make_cells(ycbcr.y.view());
    let cb_cells = make_cells(ycbcr.cb.view());
    let cr_cells = make_cells(ycbcr.cr.view());

    for cell in y_cells {
        let dc = dct_2d(cell.map(|&n| n as f64).view());
        let quantised = quantise(
            dc.view(),
            quantisation_tables::y_table_50().map(|&n| n as f64).view(),
        );
        let zigzag = zigzag(quantised.view());
    }
}

fn zigzag(array: ArrayView2<i32>) -> Array1<i32> {
    let (rows, cols) = array.dim();

    let mut output = Array1::zeros(rows * cols);
    let (mut row, mut col) = (0, 0);

    for i in 0..(rows * cols) {
        output[i] = array[(row, col)];

        if (row + col) % 2 == 0 {
            if row > 0 && col < cols - 1 {
                row -= 1;
                col += 1;
            } else if col < cols - 1 {
                col += 1;
            } else {
                row += 1;
            }
        } else {
            if col > 0 && row < rows - 1 {
                col -= 1;
                row += 1;
            } else if row < rows - 1 {
                row += 1;
            } else {
                col += 1;
            }
        }
    }

    output
}

/// Splits input array into 8x8 cells,
/// if the dimentions of the array are not multiples of 8,
/// the array is stretched to make it so
pub fn make_cells(input: ArrayView2<i8>) -> Vec<Array2<i8>> {
    let stretched = stretch(input);

    let (rows, cols) = stretched.dim();

    let cell_rows = rows / 8;
    let cell_cols = cols / 8;

    let mut cells = Vec::<Array2<i8>>::with_capacity(cell_cols * cell_rows);

    for i in 0..cell_rows {
        let start_row = i * 8;
        let end_row = start_row + 8;

        for j in 0..cell_cols {
            let start_col = j * 8;
            let end_col = start_col + 8;

            let cell = stretched
                .slice(s![start_row..end_row, start_col..end_col])
                .to_owned();
            cells.push(cell)
        }
    }

    cells
}

/// Strectes the input array so that it's dimentions are multiples of 8
fn stretch(input: ArrayView2<i8>) -> Array2<i8> {
    let (rows, cols) = input.dim();

    let stretched_rows = ((rows + 7) / 8) * 8;
    let stretched_cols = ((cols + 7) / 8) * 8;

    let mut stretched = Array2::<i8>::zeros((stretched_rows, stretched_cols));
    // Insert all of input
    stretched.slice_mut(s![..rows, ..cols]).assign(&input);

    if cols != stretched_cols {
        let last_col = input.slice(s![..rows, cols - 1]);
        for c in cols..stretched_cols {
            stretched.slice_mut(s![..rows, c]).assign(&last_col)
        }
    }

    if rows != stretched_rows {
        // Makes it too owned to stop immutable ref error
        let last_row = stretched.slice(s![rows - 1, ..]).to_owned();
        for r in rows..stretched_rows {
            stretched
                .slice_mut(s![r, ..stretched_cols])
                .assign(&last_row);
        }
    }

    stretched
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zigzag_test() {
        let input = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let expected_output = array![1, 2, 4, 7, 5, 3, 6, 8, 9];

        assert_eq!(zigzag(input.view()), expected_output)
    }

    #[test]
    fn stretch_test_1() {
        let input = array![
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ];
        let expected_output = array![
            [1, 2, 3, 4, 5, 5, 5, 5],
            [6, 7, 8, 9, 10, 10, 10, 10],
            [11, 12, 13, 14, 15, 15, 15, 15],
            [16, 17, 18, 19, 20, 20, 20, 20],
            [21, 22, 23, 24, 25, 25, 25, 25],
            [21, 22, 23, 24, 25, 25, 25, 25],
            [21, 22, 23, 24, 25, 25, 25, 25],
            [21, 22, 23, 24, 25, 25, 25, 25],
        ];
        assert_eq!(stretch(input.view()), expected_output)
    }

    #[test]
    fn stretch_test_2() {
        let input = array![
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24]
        ];

        let expected_output = array![
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [17, 18, 19, 20, 21, 22, 23, 24],
        ];
        assert_eq!(stretch(input.view()), expected_output);
    }
}
