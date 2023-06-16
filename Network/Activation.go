package Network

import (
	"ann/matrix"

	"math"
)

// this decides if the neurons on a layer will or not be activated.
// It is implemented as a node of a doubly linked list.
type Activation struct {
	Input  matrix.Matrix
	Output matrix.Matrix

	Previous *Dense
	Next     *Dense
}

// ------// Activates some neurons
func (self *Activation) Forward(X *matrix.Matrix) matrix.Matrix {
	self.Input = *X
	Y := matrix.NewMatrix(X.Lin, X.Col)

	for lin := 0; lin < Y.Lin; lin++ {
		for col := 0; col < Y.Col; col++ {
			Y.Values[lin][col] = math.Tanh(X.Values[lin][col])
		}
	}

	//------// checks if there's a layer after this.
	if self.Next != nil {
		//------// if so, activates some neurons on the next layer
		return self.Next.Forward(&Y)
	} else {
		//------// if not, returns the output of the neural network.
		return Y
	}
}

// ------// does the backward propagation
func (self *Activation) Backward(outputGradient *matrix.Matrix) matrix.Matrix {
	b := matrix.NewMatrix(self.Input.Lin, self.Input.Col)

	for lin := 0; lin < b.Lin; lin++ {
		for col := 0; col < b.Col; col++ {
			x := self.Input.Values[lin][col]
			b.Values[lin][col] = 1 - math.Pow(math.Tanh(x), 2)
		}
	}

	output, _ := matrix.Multiply(outputGradient, &b) //multiplies 2 matrices elementwise

	return self.Previous.Backward(&output)
}
