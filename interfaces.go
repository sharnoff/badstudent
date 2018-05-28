package badstudent

// import "os"

// in addition to these functions,
// each operator should be able to provide a way to deserialize its stored data,
// should the network need to be loaded from file
// the signature should be: func(*Layer, *os.File) Operator
// for more information, see Load()
type Operator interface {
	// should initialize any weights if used, and return the number of output values from the operation
	// Init() will always be run on an operator before any other method
	// Init() will only be run once
	//
	// can use *Layer.Size() to get the size of the layer
	Init(*Layer) error
	// Init(l *Layer) (int, error)

	// Should write to (and close) the file, such that the file can be used to
	// re-create the Operator.
	// the file will be purely what Serialize writes to it; nothing is added by
	// the main library itself
	// Serialize(*Layer, *os.File)

	// Should update the values of the layer to reflect the inputs and weights (if any)
	// arguments: given layer, source slice for the values of that layer
	Evaluate(*Layer, []float64) error
	// Evaluate(l *Layer, values []float64) error

	// should add to the deltas of the given range of inputs how each of those values affects the
	// total error through the values of the host layer
	//
	// arguments: given layer, a way to add to the deltas of the given input,
	// starting index of those input values, ending index of those input values
	// more details on add:
	// adds the given 'float64' to the deltas of the given input at index 'int'
	// returns error if out of bounds
	//
	// Is likely to be called in parallel for multiple inputs,
	// so this needs to be thread-safe.
	InputDeltas(*Layer, func(int, float64), int, int) error
	// InputDeltas(l *Layer, add func(int, float64), input int) error

	// returns whether or not Adjust() changes the outputs of the Layer.
	// generally will be whether or not the Layer has weights
	//
	// will be run often, but probably should not change
	// if it changes, there might be unforseen consequences
	CanBeAdjusted(*Layer) bool

	// adjusts the weights of the given layer, using its deltas
	//
	// args: layer to adjust, the learning rate to proivde the optimizer, 
	// whether or not the changes from Adjust() should be applied immediately or stored
	Adjust(*Layer, float64, bool) error
	// Adjust(l *Layer, opt Optimizer, learningRate float64, saveChanges bool) error

	// adds any changes to weights that have been delayed
	//
	// may be called without any changes waiting to happen
	AddWeights(*Layer) error
	// AddWeights(l *Layer) error
}

// optimizers should have a way to store their data, if they have any
type Optimizer interface {
	// arguments: target layer, number of weights, gradient of weight at index,
	// add to weight at index, learning rate
	//
	// number of weights can be 0
	// gradient of weights, adding to weights allows panicing
	// adding to weights is not thread-safe for repeated indexes
	Run(*Layer, int, func(int) float64, func(int, float64), float64) error
	// Run(l *Layer, size int, grad func(int) float64, add func(int, float64), learningRate float64) error
}

type CostFunction interface {
	// for all functions, can assume that length is the same, there are no NaNs or Infs, and indexes are in range

	// arguments: actual values, target values.
	Cost([]float64, []float64) (float64, error)
	// Cost(outputs, targets []float64) (float64, error)

	// should provide the derivatives of the inputs to the cost function
	// on the range [start, end), given by the two 'int's
	// args: actual values, target values, start, end, returning function
	//
	// more details on returning function:
	// args: index in given range, derivative of the total cost W.R.T. that value
	//
	// will only be run after Cost() has been run, which means that it likely won't have to re-calculate some parts
	// should NOT modify actual values or target values, as they are originals
	//
	// actual values and target values will always have the same length,
	// start and end will always be a valid range
	Deriv([]float64, []float64, int, int, func(int, float64)) error
	// Deriv(outputs, targets []float64, start, end int, returnFunc func(int, float64)) error
}
