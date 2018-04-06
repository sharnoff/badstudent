package badstudent

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