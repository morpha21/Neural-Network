package Network

import (
	"ann/matrix"
	"math"
	"math/rand"
)

// ------// this connects neurons from one layer to the next.
// ------// It is implemented as a node of a doubly linked list.
type Dense struct {
	input matrix.Matrix
	Y     matrix.Matrix

	weights matrix.Matrix
	bias    matrix.Matrix

	Previous *Activation
	Next     *Activation
}

// ------// this defines the number of neurons on the input and the output
func NewDenseLayer(inputSize, outputSize int) Dense {
	var d Dense

	d.weights = matrix.NewMatrix(outputSize, inputSize)
	d.bias = matrix.NewMatrix(outputSize, 1)

	for i := 0; i < outputSize; i++ {
		d.bias.Values[i][0] = rand.Float64()
		for j := 0; j < inputSize; j++ {
			d.weights.Values[i][j] = 1.618 * rand.Float64() * math.Pow(-1, float64(i+j))
		}
	}
	return d
}

// ------// this does the FeedForward
func (self *Dense) Forward(X *matrix.Matrix) matrix.Matrix {
	self.input = *X
	wx, _ := matrix.Product(&(self.weights), X)
	self.Y, _ = matrix.Add(&wx, &(self.bias)) // calculates the output

	return self.Next.Forward(&self.Y) // sends the output to the next layer
}

// ------// this does the backward propagation
func (self *Dense) Backward(outputGradient *matrix.Matrix) matrix.Matrix {

	//------------------//
	learningRate := 0.1 // our network's learning rate
	//------------------//

	wT := self.weights.T()
	dEdX, _ := matrix.Product(&wT, outputGradient)

	inputT := self.input.T()
	weightsGradient, _ := matrix.Product(outputGradient, &inputT)

	grad := matrix.Scale(&weightsGradient, learningRate)
	self.weights, _ = matrix.Subtract(&self.weights, &grad)

	scaledGrad := matrix.Scale(outputGradient, learningRate)
	self.bias, _ = matrix.Subtract(&self.bias, &scaledGrad)

	//----------// checks if there's a layer before this:
	if self.Previous != nil {
		//------// if so, pass the derivative of the error to the previous layer
		return self.Previous.Backward(&dEdX)
	} else {
		//------// if not, returns the derivative of the error.
		return dEdX
	}
}
