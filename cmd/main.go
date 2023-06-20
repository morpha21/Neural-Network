package main

import (
	nn "ann/Network"
	"ann/matrix"
	"fmt"
	"time"
)

func main() {

	//------------------// time measurement
	start := time.Now() //
	//------------------//

	//----------------------//
	var X [](matrix.Matrix) // each example X[i] we use to
	var Y [](matrix.Matrix) // train our network is a Matrix,
	examples := 4           // as well as each target Y[i]
	//----------------------//

	//----------------------------------------//
	for i := 0; i < examples; i++ { //--------//
		X = append(X, matrix.NewMatrix(2, 1)) // in this case, each X[i] is a column matrix (a vector)
		Y = append(Y, matrix.NewMatrix(1, 1)) // and each Y[i] is a number
	} //--------------------------------------//

	//------// We are considering the implication logical operation,
	//------// but with the given architecure (defined below),
	//------// it would work with any other binary logical operation,
	//------// such as XOR, conjunction, disjunction, etc.
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
	var epochs uint = 16180                     //
	//------------------------------------------//
	// for this particular example, this particular architecture and
	// around 10k epochs of training seems to be enough to learn the pattern.
	//

	//
	//------// Trains the neural network:
	//----------------------------//
	network.Train(&X, &Y, epochs) //
	//----------------------------//
	//------//
	//

	//
	// Finally, we get to print the prediction of each example (output.Values),
	// and the respective expected result (Y[i].Values)
	for i := 0; i < examples; i++ {
		output := network.Head.Forward(&(X[i]))
		fmt.Println(output.Values, Y[i].Values)
	}

	fmt.Println()
	duration := time.Since(start)
	fmt.Println("time elapsed:", duration)

}
