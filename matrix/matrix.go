package matrix

import (
	"errors"
	"math"
)

// ------// This is a Matrix object.
// ------// Decided to implement matrices as struct
// ------// so I could give them methods.
type Matrix struct {
	Values [][]float64
	Lin    int
	Col    int
}

// ------// builds a new matrix
func NewMatrix(lines, columns int) Matrix {

	values := make([][]float64, lines)

	//-------------------------------------//  populates the matrix
	for x := 0; x < lines; x++ {
		values[x] = make([]float64, columns)
	}

	return Matrix{Values: values, Lin: lines, Col: columns}
}

// ------// Returns the transpose of the current matrix
func (this *Matrix) T() Matrix {

	if this.Lin == 1 && this.Col == 1 {
		return *this
	}

	t := NewMatrix(this.Col, this.Lin)

	for i := 0; i < this.Lin; i++ {
		for j := 0; j < this.Col; j++ {
			t.Values[j][i] = this.Values[i][j]
		}
	}
	return t
}

// ------// Returns A+B
func Add(A, B *Matrix) (Matrix, error) {
	var err error

	if A.Lin != B.Lin || A.Col != B.Col {
		err = errors.New("matrices with different sizes")
		return *A, err
	}

	lin := A.Lin
	col := A.Col

	C := NewMatrix(lin, col)

	for i := 0; i < lin; i++ {
		for j := 0; j < col; j++ {
			C.Values[i][j] = A.Values[i][j] + B.Values[i][j]
		}
	}
	return C, err
}

// ------// Returns A-B
func Subtract(A, B *Matrix) (Matrix, error) {
	var err error

	if A.Lin != B.Lin || A.Col != B.Col {
		err = errors.New("matrices with different sizes")
		return *A, err
	}

	lin := A.Lin
	col := A.Col

	C := NewMatrix(lin, col)

	for i := 0; i < lin; i++ {
		for j := 0; j < col; j++ {
			C.Values[i][j] = A.Values[i][j] - B.Values[i][j]
		}
	}
	return C, err
}

// ------// Standard matrix product
func Product(A *Matrix, B *Matrix) (Matrix, error) {
	var err error

	if A.Col != B.Lin {
		err = errors.New("Unable to multiply matrices")
		return *A, err
	}

	C := NewMatrix(A.Lin, B.Col)

	for j := 0; j < B.Col; j++ {
		for i := 0; i < A.Lin; i++ {
			cij := 0.0
			for k := 0; k < A.Col; k++ {
				cij += A.Values[i][k] * B.Values[k][j]
			}
			C.Values[i][j] = cij
		}
	}
	return C, err
}

// ------// returns the sum of the matrix's elements
func (A *Matrix) Sum() float64 {
	lin := A.Lin
	col := A.Col

	sum := 0.0

	for i := 0; i < lin; i++ {
		for j := 0; j < col; j++ {
			sum += A.Values[i][j]
		}
	}
	return sum
}

// ------// returns the mean of the matrix's elements
func (A *Matrix) Mean() float64 {
	n := float64(A.Lin * A.Col)
	return A.Sum() / n
}

// ------// returns a matrix in which each entry is
// ------// the square of the original matrix's entry.
func Power(A *Matrix, exp float64) Matrix {
	C := NewMatrix(A.Lin, A.Col)

	for i := 0; i < A.Lin; i++ {
		for j := 0; j < A.Col; j++ {
			C.Values[i][j] = math.Pow(A.Values[i][j], exp)
		}
	}
	return C
}

// ------// returns the matrix factor*A, where `factor` is a real number
func Scale(A *Matrix, factor float64) Matrix {
	B := NewMatrix(A.Lin, A.Col)

	for i := 0; i < A.Lin; i++ {
		for j := 0; j < A.Col; j++ {
			B.Values[i][j] = factor * A.Values[i][j]
		}
	}
	return B
}

// ------// multiply A and B elementwise.
func Multiply(A, B *Matrix) (Matrix, error) {
	var err error

	if A.Lin != B.Lin || A.Col != B.Col {
		err = errors.New("matrices with different sizes")
		return *A, err
	}

	C := NewMatrix(A.Lin, A.Col)

	for i := 0; i < A.Lin; i++ {
		for j := 0; j < A.Col; j++ {
			C.Values[i][j] = A.Values[i][j] * B.Values[i][j]
		}
	}
	return C, err
}

// ------// returns the minimum element of A
func (A *Matrix) Min() float64 {
	min := A.Values[0][0]
	for i := 0; i < A.Lin; i++ {
		for j := 0; j < A.Col; j++ {
			if A.Values[i][j] < min {
				min = A.Values[i][j]
			}
		}
	}
	return min
}

// ------// returns the maximum element of A
func (A *Matrix) Max() float64 {
	max := A.Values[0][0]
	for i := 0; i < A.Lin; i++ {
		for j := 0; j < A.Col; j++ {
			if A.Values[i][j] > max {
				max = A.Values[i][j]
			}
		}
	}
	return max
}
