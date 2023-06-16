package main

import (
	nn "ann/Network"
	Losses "ann/losses"
	"ann/matrix"
	"fmt"
)

func main() {

	//-----------------------//   each example X[i] we use
	var X [4](matrix.Matrix) // to train our network is a Matrix,
	var Y [4](matrix.Matrix) // as well as each target Y[i]
	examples := len(X)
	//-----------------------//
	//

	//
	//--------------------------------//
	for i := 0; i < examples; i++ { ////
		X[i] = matrix.NewMatrix(2, 1) // in this case, each X[i] is a column matrix (a vector)
		Y[i] = matrix.NewMatrix(1, 1) // and each Y[i] is a number
	} //------------------------------//

	// ------ // We are considering the implication logical operation,
	// ------ // but with the given architecure, it would work with
	// ------ // any other binary logical operation, such as
	// ------ // XOR, conjunction, disjunction, etc.
	//----------------------------------//
	X[0].Values = [][]float64{{0}, {0}} //
	Y[0].Values = [][]float64{{1}}      //
	X[1].Values = [][]float64{{0}, {1}} //
	Y[1].Values = [][]float64{{1}}      //
	X[2].Values = [][]float64{{1}, {0}} //
	Y[2].Values = [][]float64{{0}}      //
	X[3].Values = [][]float64{{1}, {1}} //
	Y[3].Values = [][]float64{{1}}      //
	//----------------------------------//

	// We are now defining the Neural Network architecture, giving it 2 layers:
	// one receiving 2 numbers and spitting out 3 to the next layer;
	// and the other receiving 3 numbers and spitting 1 as an output.
	//------------------------------------------//
	network, _ := nn.NewNetwork([]int{2, 3, 1}) //
	epochs := 16180                             //
	//------------------------------------------//
	// for this particular example, this particular architecture and
	// around 10k epochs of training seems to be enough to learn the pattern.
	//
	//

	//
	//
	//-------------------------------------------------------//
	//	Here, we are training the network `epochs` times,
	//	and also calculating and printing the error.
	//-------------------------------------------------------//
	for epoch := 0; epoch < epochs; epoch++ {
		errorE := 0.0
		for i := 0; i < examples; i++ {
			//-------------------------------// "network.Head" is the first dense layer of the Network,
			y := network.Head.Forward(&X[i]) // and "y" receives the calculated output of the i-th
			//-------------------------------// training example.

			errorE += Losses.MeanSquaredError(&Y[i], &y)

			//----------------------------------------------// "network.Tail" is the last activation layer
			grad := Losses.MeanSquaredErrorPrime(&Y[i], &y) // of the Network, and here we're doing the
			grad = network.Tail.Backward(&grad)             // backpropagation, updating the network
			//----------------------------------------------// weights and biases.
		}
		errorE = errorE / float64(examples)
		fmt.Printf("%d/%d, error = %f\n", epoch+1, epochs, errorE) // prints the error on each epoch
	}
	//-------------------------------------------------------------//
	//-------------------------------------------------------------//
	//

	//
	// Finally, we get to print the prediction of each example (output.Values),
	// and the respective expected result (Y[i].Values)
	for i := 0; i < examples; i++ {
		output := network.Head.Forward(&(X[i]))
		fmt.Println(output.Values, Y[i].Values)
	}
	fmt.Println()

}
