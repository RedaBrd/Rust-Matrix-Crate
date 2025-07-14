// Matrix library for basic matrix operations (3x3 only).
// Construction is enforced via MatrixBuilder for safety.

mod matrix {
    use std::{
        fmt::{Debug, Display},
        ops::{Add, Div, Mul, Neg, Sub},
    };

    /// Error types for matrix operations.
    #[derive(Debug)]
    pub enum MatrixErrors {
        IncompatibleDimensions,
        InverseOfZeroDeterminant,
        DeterminantOfNonSquare,
    }

    /// Row wrapper for matrix dimensions.
    #[derive(PartialEq, Copy, Clone, Debug)]
    pub struct Row(pub usize);

    /// Column wrapper for matrix dimensions.
    #[derive(PartialEq, Copy, Clone, Debug)]
    pub struct Column(pub usize);

    /// Data wrapper for matrix elements.
    #[derive(Debug, PartialEq)]
    struct Data<T>(Vec<T>);

    /// Matrix struct, only constructible via MatrixBuilder.
    #[derive(Debug, PartialEq)]
    pub struct Matrix<T> {
        row: Row,
        column: Column,
        data: Data<T>,
    }

    /// Builder for safe matrix construction.
    #[derive(Debug)]
    pub struct MatrixBuilder<T> {
        row: Option<Row>,
        column: Option<Column>,
        data: Option<Data<T>>,
    }


    impl<T> Matrix<T>
    where
        T: PartialEq
            + PartialOrd
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Mul<Output = T>
            + Display
            + Debug
            + Default
            + Copy
            + Clone
            + Neg<Output = T>,
    {
        /// Compute the determinant of a 3x3 matrix.
        pub fn determinant3x3(&self) -> Result<T, MatrixErrors> {
            if self.row.0 == 3 && self.column.0 == 3 {
                Ok(
                    self.data.0[0] * self.data.0[4] * self.data.0[8]
                        + self.data.0[3] * self.data.0[7] * self.data.0[2]
                        + self.data.0[6] * self.data.0[1] * self.data.0[5]
                        - self.data.0[2] * self.data.0[4] * self.data.0[6]
                        - self.data.0[1] * self.data.0[3] * self.data.0[8]
                        - self.data.0[5] * self.data.0[7] * self.data.0[0],
                )
            } else {
                Err(MatrixErrors::IncompatibleDimensions)
            }
        }

        /// Return the transpose of the matrix.
        pub fn transpose(&self) -> Self {
            let mut res = vec![T::default(); self.data.0.len()];
            let m = self.row.0;
            let n = self.column.0;
            for i in 0..m {
                for j in 0..n {
                    res[j * m + i] = self.data.0[i * n + j];
                }
            }
            Matrix {
                row: Row(self.column.0),
                column: Column(self.row.0),
                data: Data(res),
            }
        }

        /// Compute the inverse of a 3x3 matrix.
        pub fn inverse3x3(&self) -> Result<Self, MatrixErrors> {
            if self.row.0 != 3 || self.column.0 != 3 {
                return Err(MatrixErrors::IncompatibleDimensions);
            }
            let det = self.determinant3x3()?;
            if det == det - det {
                return Err(MatrixErrors::InverseOfZeroDeterminant);
            }
            let adj = self.adjugate3x3()?;
            Ok(adj.scalar_multiplication((det / det) / det))
        }

        /// Multiply all elements by a scalar.
        pub fn scalar_multiplication(mut self, r: T) -> Self {
            for item in &mut self.data.0 {
                *item = r * *item;
            }
            self
        }

        /// Compute the adjugate of a 3x3 matrix.
        pub fn adjugate3x3(&self) -> Result<Self, MatrixErrors> {
            if self.row.0 != 3 || self.column.0 != 3 {
                return Err(MatrixErrors::IncompatibleDimensions);
            }
            let m = &self.data.0;
            let get = |r: usize, c: usize| m[r * 3 + c];

            let cof00 = get(1, 1) * get(2, 2) - get(1, 2) * get(2, 1);
            let cof01 = -(get(1, 0) * get(2, 2) - get(1, 2) * get(2, 0));
            let cof02 = get(1, 0) * get(2, 1) - get(1, 1) * get(2, 0);

            let cof10 = -(get(0, 1) * get(2, 2) - get(0, 2) * get(2, 1));
            let cof11 = get(0, 0) * get(2, 2) - get(0, 2) * get(2, 0);
            let cof12 = -(get(0, 0) * get(2, 1) - get(0, 1) * get(2, 0));

            let cof20 = get(0, 1) * get(1, 2) - get(0, 2) * get(1, 1);
            let cof21 = -(get(0, 0) * get(1, 2) - get(0, 2) * get(1, 0));
            let cof22 = get(0, 0) * get(1, 1) - get(0, 1) * get(1, 0);

            let adjugate = vec![
                cof00, cof10, cof20,
                cof01, cof11, cof21,
                cof02, cof12, cof22,
            ];

            Ok(Matrix {
                row: Row(3),
                column: Column(3),
                data: Data(adjugate),
            })
        }
    }

    impl<T> MatrixBuilder<T>
    where
        T: PartialEq
            + PartialOrd
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Mul<Output = T>
            + Display
            + Debug,
    {
        /// Create a new builder.
        pub fn new() -> Self {
            MatrixBuilder {
                row: None,
                column: None,
                data: None,
            }
        }

        /// Set the number of rows.
        pub fn row(&mut self, row: Row) {
            self.row = Some(row);
        }

        /// Set the number of columns.
        pub fn column(&mut self, column: Column) {
            self.column = Some(column);
        }

        /// Set the data for the matrix.
        pub fn data(&mut self, data: Vec<T>) {
            if let (Some(row), Some(column)) = (&self.row, &self.column) {
                if data.len() == row.0 * column.0 {
                    self.data = Some(Data(data));
                }
            }
        }

        /// Build the matrix if all fields are set and valid.
        pub fn build(self) -> Option<Matrix<T>> {
            Some(Matrix {
                row: self.row?,
                column: self.column?,
                data: self.data?,
            })
        }
    }

    // Arithmetic operations for matrices.
    mod matrix_arithmetic {
        use std::ops::AddAssign;

        use super::*;
        /// Matrix addition.
        impl<'a, T> Add for &'a Matrix<T>
        where
            &'a T: Add<&'a T, Output = T> + 'a,
            T: PartialEq
                + PartialOrd
                + Add<Output = T>
                + Sub<Output = T>
                + Div<Output = T>
                + Mul<Output = T>
                + Display
                + Debug,
        {
            type Output = Result<Matrix<T>, MatrixErrors>;

            fn add(self, rhs: Self) -> Self::Output {
                if self.column == rhs.column && self.row == rhs.row {
                    let res: Vec<T> = self.data.0.iter()
                        .zip(&rhs.data.0)
                        .map(|(a, b)| a + b)
                        .collect();
                    Ok(Matrix {
                        row: self.row,
                        column: self.column,
                        data: Data(res),
                    })
                } else {
                    Err(MatrixErrors::IncompatibleDimensions)
                }
            }
        }
        /// Matrix subtraction.
        impl<'a, T> Sub for &'a Matrix<T>
        where
            &'a T: Sub<&'a T, Output = T> + 'a,
            T: PartialEq
                + PartialOrd
                + Add<Output = T>
                + Sub<Output = T>
                + Div<Output = T>
                + Mul<Output = T>
                + Display
                + Debug,
        {
            type Output = Result<Matrix<T>, MatrixErrors>;

            fn sub(self, rhs: Self) -> Self::Output {
                if self.column == rhs.column && self.row == rhs.row {
                    let res: Vec<T> = self.data.0.iter()
                        .zip(&rhs.data.0)
                        .map(|(a, b)| a - b)
                        .collect();
                    Ok(Matrix {
                        row: self.row,
                        column: self.column,
                        data: Data(res),
                    })
                } else {
                    Err(MatrixErrors::IncompatibleDimensions)
                }
            }
        }

        /// Matrix multiplication.
        impl<'a, T> Mul for &'a Matrix<T>
        where
            &'a T: Mul<&'a T, Output = T> + 'a,
            T: PartialEq
                + PartialOrd
                + Add<Output = T>
                + Sub<Output = T>
                + Div<Output = T>
                + Mul<Output = T>
                + Display
                + Debug
                + Default
                + Clone
                + AddAssign
                + Copy,
        {
            type Output = Result<Matrix<T>, MatrixErrors>;

            fn mul(self, rhs: Self) -> Self::Output {
                if self.column.0 == rhs.row.0 {
                    let mut res = vec![T::default(); self.row.0 * rhs.column.0];
                    for i in 0..self.row.0 {
                        for j in 0..rhs.column.0 {
                            let mut sum = T::default();
                            for k in 0..self.column.0 {
                                sum += self.data.0[i * self.column.0 + k]
                                    * rhs.data.0[k * rhs.column.0 + j];
                            }
                            res[i * rhs.column.0 + j] = sum;
                        }
                    }
                    Ok(Matrix {
                        row: self.row,
                        column: rhs.column,
                        data: Data(res),
                    })
                } else {
                    Err(MatrixErrors::IncompatibleDimensions)
                }
            }
        }
    }

    // Display implementation for pretty-printing matrices.
    mod output {
        use std::fmt::{Debug, Display, Formatter};
        use super::*;
        impl<T> Display for Matrix<T>
        where
            T: Debug,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                let mut res = String::new();
                for i in 0..self.data.0.len() {
                    if i % self.column.0 == 0 {
                        res.push('\n');
                    }
                    res.push_str(&format!("{:?}", self.data.0[i]));
                }
                write!(f, "{}", res)
            }

            
        }
    }

    // Iterator implementation for Data.
    mod iter {
        use crate::matrix::Matrix;


        impl<T> IntoIterator for Matrix<T> {
            type Item = T;
            type IntoIter = std::vec::IntoIter<T>;

            fn into_iter(self) -> Self::IntoIter {
                self.data.0.into_iter()
            }
        }
    }
}
