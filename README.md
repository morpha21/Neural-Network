# Dense Neural Network in Go

This is a Dense Neural Network written in Go for learning purposes. It currently works 
for predicting binary logical operations, such as logical implication and XOR. To 
define training examples, use the `x.csv` and `y.csv` files for features and target 
variables, respectively. Ensure that the CSV files do not have headers. The 
architecture of the Artificial Neural Network (ANN) is currently hard-coded in an 
array, and the first and last numbers in it must match the number of columns in both 
`x.csv` and `y.csv` files, respectively.

## Implementation Details

The Neural Network is implemented in a Doubly Linked List fashion, with activation and 
dense layers implemented as nodes of the list. The Network itself is a struct with a 
Head pointer to the first node (a Dense layer) and a Tail pointer pointing to the last 
node (an Activation layer).

### Dense Layers

Dense layers are structs with pointers to the next and previous (if it exists) 
activation layers. They also have methods that connect them to those layers.

### Activation Layers

Activation layers are structs with pointers to the previous and next (if exists) dense 
layers. They have methods similar to those found in Dense layers. Currently, the ANN 
is using only `tanh` as activation function.
