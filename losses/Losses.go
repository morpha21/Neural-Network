package losses

import (
	"ann/matrix"
)

// ------// calculates the Mean Squared Error between
// ------// predicted output and training values.
func MeanSquaredError(y *(matrix.Matrix), pred *(matrix.Matrix)) float64 {
	dif, _ := matrix.Subtract(y, pred)

	sqDif := matrix.Power(&dif, 2)

	return sqDif.Mean()
}

// ------// calculates the above's derivative
func MeanSquaredErrorPrime(y, pred *(matrix.Matrix)) matrix.Matrix {
	dif, _ := matrix.Subtract(pred, y)

	dif2 := matrix.Scale(&dif, 2/float64(len(y.Values)))
	return dif2
}
