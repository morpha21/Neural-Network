package main

import (
	nn "ann/Network"
	"ann/utils"
	"fmt"
	"os"
	"time"
)

func main() {

	//------------------// time measurement
	start := time.Now() //
	//------------------//

	x_file := os.Args[1]
	y_file := os.Args[2]

	X := utils.ReadCSV(x_file)
	Y := utils.ReadCSV(y_file)
	fmt.Println()

	// We are now defining the Neural Network architecture, giving it 2 layers:
	// one receiving 2 numbers and spitting out 3 to the next layer;
	// and the other receiving 3 numbers and spitting 1 as an output.
	//-------------------------?-----------------//
	network, _ := nn.NewNetwork([]int{2, 3, 1}) //
	var epochs uint = 8128                      //
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
	for i := 0; i < len(X); i++ {
		output := network.Head.Forward(&(X[i]))
		fmt.Println(output.Values, Y[i].Values)
	}

	fmt.Println()
	duration := time.Since(start)
	fmt.Println("time elapsed:", duration)

	fmt.Println()
}
