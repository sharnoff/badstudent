package badstudent

import (
	"sort"
	"sync"
)

type Layer struct {
	// The name that will be used to print this layer.
	// Not used for unique identification of any kind,
	// and may be nil
	Name string

	// used for validation during setup
	hostNetwork *Network

	// the values of the layer -- essentially its outputs
	values  []float64

	// the derivative of each value w.r.t. the total cost
	// of the particular training example
	deltas  []float64 // δ

	// soon to be removed
	weights [][]float64

	// the layers that the given layer inputs from
	// could be nil
	inputs  []*Layer

	// for each layer, the sum of the number of its values and those
	// of layers in previous indexes
	// ex: the index for the last input will have the total
	// number of input values to the layer
	numInputs []int

	// the layers that the given layer outputs to
	// can be nil. Can also be non-nil and layer is an output
	outputs []*Layer

	// whether or not the layer's values are part of the set of
	// output values to the network
	isOutput bool

	// what index in the network outputs the values of the layer start at
	// ex: for the first output layer, its 'placeInOutputs' would be 0
	placeInOutputs int

	// keeps track of what has been calculated or completed for the layer
	status status_

	// a lock for 'status'
	statusMux sync.Mutex
}

// returns the number of values that the layer has
func (l *Layer) Size() int {
	return len(l.values)
}

// returns the value of the input to the layer at that index
//
// allows panics from index out of bounds and nil pointer
// a nil pointer means that the layer has no inputs
func (l *Layer) InputValue(index int) float64 {
	greaterThan := func(i int) bool {
		return index < l.numInputs[i]
	}

	i := sort.Search(l.numInputs[len(l.inputs) - 1], greaterThan)

	return l.inputs[i].values[ len(l.inputs[i].values) - index + l.numInputs[i] ]
}

// returns the number of layers that the layer receives input from
func (l *Layer) NumInputLayers() int {
	return len(l.inputs)
}

// returns the total number of input values to the layer
func (l *Layer) NumInputs() int {
	if len(l.inputs) == 0 {
		return 0
	}

	return l.numInputs[len(l.inputs) - 1]
}

// Returns the number of values that provide input to the layer
// before the layer at the given index does
// Can provide index of *Layer.NumInputLayers(), which will return total number of inputs
//
// will allow the index out of bounds panic to go through if index is not in range
func (l *Layer) PreviousInputs(index int) int {
	if index == 0 {
		return 0
	} else {
		return l.numInputs[index - 1]
	}
}

// returns a single slice containing a copy of all of
// the input values to the layer, in order
func (l *Layer) CopyOfInputs() []float64 {
	
	c := make([]float64, l.NumInputs())
	start := 0
	for _, in := range l.inputs {
		copy(c[start : start + in.Size()], in.values)
		start += in.Size()
	}

	return c
}