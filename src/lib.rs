mod matrix {
    use std::{
        fmt::{Debug, Display},
        ops::{Add, Div, Mul, Neg, Sub},
    };
    #[derive(Debug)]
    pub enum MatrixErrors {
        IncompatibleDimensions,
        InverseOfZeroDeterminant,
        DeterminantOfNonSquare,
    }

    #[derive(PartialEq, Copy, Clone, Debug)]
    pub struct Row(pub usize);

    #[derive(PartialEq, Copy, Clone, Debug)]
    pub struct Column(pub usize);

    #[derive(Debug)]
    pub struct Data<T>(pub Vec<T>);

    #[derive(Debug)]
    pub struct Matrix<T> {
        pub row: Row,
        pub column: Column,
        pub data: Data<T>,
    }

    #[derive(Debug)]
    struct MatrixBuilder<T> {
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

        pub fn inverse3x3(&self) -> Result<Self, MatrixErrors> {
            if self.row.0 != 3 || self.column.0 != 3 {
                return Err(MatrixErrors::IncompatibleDimensions);
            }
            if let Ok(r) = self.determinant3x3() {
                if let Ok(mut m) = self.adjugate3x3() {
                    if r != r - r {
                        m.scalar_multiplication((r / r) / r);
                        return Ok(m);
                    } else {
                        return Err(MatrixErrors::InverseOfZeroDeterminant);
                    }
                } else {
                    return Err(MatrixErrors::IncompatibleDimensions);
                }
            } else {
                return Err(MatrixErrors::IncompatibleDimensions);
            }
        }

        pub fn scalar_multiplication(&mut self, r: T) {
            for item in &mut self.data.0 {
                *item = r * *item;
            }
        }

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
                cof00, cof10, cof20, //
                cof01, cof11, cof21, //
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
        pub fn new() -> Self {
            MatrixBuilder {
                row: None,
                column: None,
                data: None,
            }
        }

        pub fn row(&mut self, row: Row) {
            self.row = Some(row);
        }

        pub fn column(&mut self, column: Column) {
            self.column = Some(column);
        }

        pub fn data(&mut self, data: Vec<T>) {
            match &self.row {
                Some(row) => match &self.column {
                    Some(column) => {
                        if data.len() != row.0 * column.0 {
                            return;
                        } else {
                            self.data = Some(Data(data));
                        }
                    }
                    None => return,
                },
                None => return,
            };
        }

        pub fn build(self) -> Option<Matrix<T>> {
            match self.data {
                Some(data) => Some(Matrix {
                    row: self.row?,
                    column: self.column?,
                    data,
                }),
                None => None,
            }
        }
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
            + Debug,
    {
    }

    mod matrix_arithmetic {
        use std::ops::AddAssign;

        use super::*;

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
                    let mut res: Vec<T> = vec![];
                    let size = self.data.0.len();
                    for i in 0..size {
                        res.push(&self.data.0[i] + &rhs.data.0[i]);
                    }
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
                    let mut res = vec![];
                    let size = self.data.0.len();
                    for i in 0..size {
                        res.push(&self.data.0[i] - &rhs.data.0[i]);
                    }
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
                    let mut res: Vec<T> = vec![T::default(); self.row.0 * rhs.column.0];
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

    mod output {
        use std::fmt::Debug;
        use super::*;

        impl<T> Display for Matrix<T>
        where
            T: Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut res = String::new();

                for i in 0..self.data.0.len() {
                    if i % self.column.0 == 0 {
                        res.push_str("\n");
                    }
                    res.push_str(&format!("{:?}", self.data.0[i]));
                }
                write!(f, "{}", res)
            }
        }
    }

    mod equality {
        use super::*;

        impl<T> PartialEq for Data<T>
        where
            T: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<T> PartialEq for Matrix<T>
        where
            T: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.data == other.data
            }
        }
    }

    mod iter {
        use crate::matrix::Data;

        impl<T> IntoIterator for Data<T> {
            type Item = T;
            type IntoIter = std::vec::IntoIter<T>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::matrix::{Matrix, Row, Column, Data};

    fn matrix3x3(data: Vec<f64>) -> Matrix<f64> {
        Matrix {
            row: Row(3),
            column: Column(3),
            data: Data(data),
        }
    }

    #[test]
    fn test_determinant3x3() {
        let m = matrix3x3(vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0,
        ]);
        let det = m.determinant3x3().unwrap();
        assert_eq!(det, 1.0 * (1.0 * 0.0 - 4.0 * 6.0) - 2.0 * (0.0 * 0.0 - 4.0 * 5.0) + 3.0 * (0.0 * 6.0 - 1.0 * 5.0));
        assert_eq!(det, 1.0);
    }

    #[test]
    fn test_transpose3x3() {
        let m = matrix3x3(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let t = m.transpose();
        assert_eq!(t.data.0, vec![
            1.0, 4.0, 7.0,
            2.0, 5.0, 8.0,
            3.0, 6.0, 9.0,
        ]);
    }

    #[test]
    fn test_adjugate3x3() {
        let m = matrix3x3(vec![
            3.0, 0.0, 2.0,
            2.0, 0.0, -2.0,
            0.0, 1.0, 1.0,
        ]);
        let adj = m.adjugate3x3().unwrap();
        assert_eq!(adj.data.0, vec![
            2.0, 2.0, 0.0,
            -2.0, 3.0, 10.0,
            2.0, -3.0, 0.0,
        ]);
    }

    #[test]
    fn test_inverse3x3() {
        let m = matrix3x3(vec![
            3.0, 0.0, 2.0,
            2.0, 0.0, -2.0,
            0.0, 1.0, 1.0,
        ]);
        let inv = m.inverse3x3().unwrap();
        let expected = vec![
            0.2, 0.2, 0.0,
            -0.2, 0.3, 1.0,
            0.2, -0.3, 0.0,
        ];
        for (a, b) in inv.data.0.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_sub() {
        let a = matrix3x3(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let b = matrix3x3(vec![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ]);

        let add = (&a + &b).unwrap();
        let sub = (&a - &b).unwrap();

        assert_eq!(add.data.0, vec![
            10.0, 10.0, 10.0,
            10.0, 10.0, 10.0,
            10.0, 10.0, 10.0,
        ]);

        assert_eq!(sub.data.0, vec![
            -8.0, -6.0, -4.0,
            -2.0, 0.0, 2.0,
            4.0, 6.0, 8.0,
        ]);
    }

    #[test]
    fn test_mul3x3() {
        let a = matrix3x3(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let b = matrix3x3(vec![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ]);
        let result = (&a * &b).unwrap();

        assert_eq!(result.data.0, vec![
            30.0, 24.0, 18.0,
            84.0, 69.0, 54.0,
            138.0, 114.0, 90.0,
        ]);
    }

    #[test]
    fn test_inverse_of_zero_determinant() {
        let singular = matrix3x3(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let inv = singular.inverse3x3();
        assert!(inv.is_err());
    }
}
