package smartlearning

import ()

// calcSteps are used in order to keep track of which elements of a neuron have been calculated, and which have not
// this prevents unsafe operations, such as calculating how much the weights need to change when the value hasn't been calculated
type calcStep int8

const (
	// validOutputs calcStep = -1, used in *Network.SetOutputs()
	inputsChanged  calcStep = 0
	evaluated      calcStep = 1
	delta          calcStep = 2
	weightsChanged calcStep = 3
)
