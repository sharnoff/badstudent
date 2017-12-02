package smartlearning

import ()

type SegmentType struct {
	// dims of inputs, num values for each input, dims : num values, num weights, error
	// should check if the dimensions are valid, and return an error if not
	NumValuesAndWeights func([][]int, []int, []int) (int, int, error)

	// dims of inputs, input values, dims, weights, values (slice to fill) : error
	Calculate func([][]int, []float64, []int, []float64, []float64) error

	// inputs, target segment, input dims, input values, dims, weights, current deltas : new deltas for input, error
	InputDeltas func([]*Segment, *Segment, [][]int, []float64, []int, []float64, []float64) ([]float64, error)

	// weights, deltas, learning rate : error
	Adjust func([]float64, []float64, float64) error
}

// @IN_PROGRESS
// var FeedForwardLayer_Tanh SegmentType {
// 	NumValuesAndWeights: func (inputDims [][]int, numInputValues []int, dims []int) (int, int, error) {
// 		// inputDims doesn't actually matter
// 		if len(dims) != 1 { // @CONSTANT - this is just a feature of how 

// 		}

// 	},
// 	Calculate: func([][]int, []float64, []int, []float64, []float64) error




// }