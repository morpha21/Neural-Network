# Neural-Network
This is a Dense Neural Network written in Go, for learning purposes. 
It is currently working for predicting binary logical operations, 
like the logical implication and XOR. The training examples are
supposed to bedefined in the main file, as well as the ANN architecure. 

The Neural Network is currently implemented in a Doubly Linked List Fashion, 
with activation and dense layers being implemented as nodes of the list. 
The Network itself is a struct, with a Head pointer to the first node, 
a Dense layer, and a Tail pointer pointing to the last node, an
Activation layer. 

Dense layers are structs, with pointers to the next and previous(if exists)
activation layers, and methods that connects them to those layers. 
Activation layers are structs, with pointers to the previous and 
next(if exists) dense layers, and methods similar to those on Dense layers. 
