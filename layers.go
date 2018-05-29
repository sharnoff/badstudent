package badstudent

import (
	"sort"
	"sync"
	// "fmt"
)

type Layer struct {
	// The name that will be used to print this layer.
	// Used for unique identification, can be empty
	Name string

	// used for order identification of which layers were added first
	id int

	// used for validation during setup
	hostNetwork *Network

	// handles all of the actual operations from
	typ Operator

	// the values of the layer -- essentially its outputs
	values []float64

	// the derivative of each value w.r.t. the total cost
	// of the particular training example
	deltas []float64 // δ

	// separate from 'status.' true iff inputDeltas() have been run on outputs to layer
	deltasActuallyCalculated bool

	// the layers that the given layer inputs from
	// could be nil
	inputs []*Layer

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

// returns the the Name of the Layer, surrounded by double quotes
func (l *Layer) String() string {
	return "\"" + l.Name + "\""
}

// returns the number of values that the layer has
func (l *Layer) Size() int {
	return len(l.values)
}

// returns the dimensions of the layer, as provided by its Operator
func (l *Layer) Dimensions() []int {
	d := l.typ.Dimensions(l)
	if d == nil {
		return []int{l.Size()}
	}
	return d
}

func (l *Layer) Value(index int) float64 {
	return l.values[index]
}

// returns the value of the input to the layer at that index
//
// allows panics from index out of bounds and nil pointer
// a nil pointer means that the layer has no inputs
//
// binary searches to find the value
func (l *Layer) InputValue(index int) float64 {
	greaterThan := func(i int) bool {
		return index < l.numInputs[i]
	}

	i := sort.Search(len(l.inputs), greaterThan)

	if i > 0 {
		index -= l.numInputs[i-1]
	}

	return l.inputs[i].values[index]
}

// returns an unbuffered channel that goes through each value of the inputs
func (l *Layer) InputIterator() chan float64 {
	ch := make(chan float64)
	go func() {
		for _, in := range l.inputs {
			for _, v := range in.values {
				ch <- v
			}
		}

		close(ch)
	}()
	return ch
}

// returns the 'delta' of the value at of the layer at the given index
//
// this is a SIMPLE FUNCTION. does not check if deltas have
// been calculated before running.
func (l *Layer) Delta(index int) float64 {
	return l.deltas[index]
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

	return l.numInputs[len(l.inputs)-1]
}

// returns the size of the given input to the layer
func (l *Layer) InputSize(index int) int {
	return l.inputs[index].Size()
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
		return l.numInputs[index-1]
	}
}

// returns a single slice containing a copy of all of
// the input values to the layer, in order
func (l *Layer) CopyOfInputs() []float64 {

	c := make([]float64, l.NumInputs())
	start := 0
	for _, in := range l.inputs {
		copy(c[start:start+in.Size()], in.values)
		start += in.Size()
	}

	return c
}
