package Network

import (
	Losses "ann/losses"
	"ann/matrix"
	"errors"
	"fmt"
)

// ------//	The network is structured in a Doubly Linked List fashion,
// ------//	where the nodes are Dense Layers and Activation Layers.
type Network struct {
	Head             *Dense      //	first layer of the network
	Tail             *Activation //	last layer of the network
	DenseLayers      []Dense
	ActivationLayers []Activation
}

// This builds a new Dense Neural Network.
// `neurons` is a slice defining the number of
// neurons and layers. If neurons == [2, 3, 1],
// then it will have 3 layers,
// 2 neurons on the initial layer,
// 3 in the hidden layer, and
// 1 neuron in the output layer.
func NewNetwork(neurons []int) (Network, error) {
	var err error

	var Net Network

	// error handling
	if len(neurons) < 2 {
		err = errors.New("Unable to build network. Check the network's layers and neurons")
		return Net, err
	}

	layers := len(neurons) - 1

	Net.DenseLayers = make([]Dense, layers)
	Net.ActivationLayers = make([]Activation, layers)

	for i := 0; i < layers; i++ {
		Net.DenseLayers[i] = NewDenseLayer(neurons[i], neurons[i+1])

		Net.DenseLayers[i].Next = &Net.ActivationLayers[i]
		Net.ActivationLayers[i].Previous = &Net.DenseLayers[i]

		if i > 0 {
			Net.DenseLayers[i].Previous = &Net.ActivationLayers[i-1]
		}
		if i < layers-1 {
			Net.ActivationLayers[i].Next = &Net.DenseLayers[i+1]
		}
	}

	Net.Head = &Net.DenseLayers[0]
	Net.Tail = &Net.ActivationLayers[layers-1]

	return Net, err
}

// trains the network
func (N *Network) Train(inputs, outputs *([]matrix.Matrix), epochs uint) {
	examples := len(*inputs)
	var epoch uint

	var errorE float64
	for epoch = 0; epoch < epochs; epoch++ {
		errorE = 0.0
		for i := 0; i < examples; i++ {
			//--------------------------------------// "network.Head" is the first dense layer of the
			y_pred := N.Head.Forward(&(*inputs)[i]) // Network, and "y_pred" receives the calculated
			//--------------------------------------// output of the i-th training example.

			errorE += Losses.MeanSquaredError(&(*outputs)[i], &y_pred)

			//------------------------------------------------------------//
			grad := Losses.MeanSquaredErrorPrime(&(*outputs)[i], &y_pred) // Here we're doing the backpropagation,
			grad = N.Tail.Backward(&grad)                                 // updating the network weights and biases.
			//------------------------------------------------------------//
		}
		errorE = errorE / float64(examples)
		// fmt.Printf("%d/%d, error = %f\n\n", epoch+1, epochs, errorE) // prints the error on each epoch

	}
	fmt.Printf("error = %f\n\n", errorE) // prints the error on each epoch
}
