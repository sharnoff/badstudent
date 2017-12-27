package smartlearning

import (
	"github.com/pkg/errors"
	"math"
	"math/rand"
)

/****************************
contains the types:
 * Blank - can only be used for inputs
 * Softmax
 * Tanh
 * Neurons

Those without weights:
 	Those that preserve dimensions:
 		* Blank
 		* Softmax
 		* Tanh
 	Those that change dimensions:
Those with weights:
	Those that preserve dimensions:
	Those that change dimensions:
		* Neurons
****************************/
type SegmentType interface {
	// args: inputDims, numVpI, numInputs, dims --> returns: numValues, numWeights
	NumValuesAndWeights([][]int, []int, int, []int) (int, int, error)
	// args: weights, inputDims, numVpI, numInputs, dims
	InitWeights([]float64, [][]int, []int, int, []int) (error)

	// args: numVpI, inputDims, inVals, weights, values, deltas --> returns: (*Segment).inputDeltas()
	InputDeltasFunc([]int, [][]int, []float64, []float64, []float64, []float64) (func(int, []float64) error, error)
	// args: inVals, weights, inputDims, dims, values --> returns: (*Segment).calculate()
	CalculateFunc([]float64, []float64, [][]int, []int, []float64) (func() error, error)
	// args: deltas, weights, inVals
	AdjustFunc([]float64, []float64, []float64) (func(float64) error, error)
}

func deltasHelper(ind int, numVpI []int, inVals []float64) (int, error) {
	start := 0
	i := 0
	for i, v := range numVpI {
		if i == ind {
			break
		}
		start += v
	}

	if start+numVpI[i] > len(inVals) {
		return 0, errors.Errorf("can't get input deltas: start + numVpI[i] >= len(inVals) (%d + %d >= %d)", start, numVpI[i], len(inVals))
	}

	return start, nil
}

// Blank does nothing, and should be used exclusively for network inputs
type blank int8

func Blank() blank {
	return blank(0)
}

func (t blank) NumValuesAndWeights(inputDims [][]int, numVpI []int, numValues int, dims []int) (int, int, error) {
	if len(inputDims) != 0 || len(numVpI) != 0 || numValues != 0 {
		return 0, 0, errors.Errorf("can't give num values/weights: Segment has inputs. Blank should be used exclusively for network inputs. (inputDims: %v, numVpI: %v, numValues: %v)", inputDims, numVpI, numValues)
	} else if dims[0] <= 0 {
		return 0, 0, errors.Errorf("can't give num values <= 0 (dims[0] <= 0, dims[0] = %d)", dims[0])
	}

	return dims[0], 0, nil
}
func (t blank) InitWeights(weights []float64, inputDims [][]int, numVpI []int, numInputs int, dims []int) error {
	if len(weights) != 0 || len(inputDims) != 0 || len(numVpI) != 0 || numInputs != 0 {
		return errors.Errorf("can't initialize weights: Segment has inputs. Blank should be used exclusively for network inputs. (weights: %v, inputDims: %v, numVpI: %v, numInputs: %v", weights, inputDims, numVpI, numInputs)
	}

	return nil
}
func (t blank) InputDeltasFunc(numVpI []int, inputDims [][]int, inVals, weights, values, deltas []float64) (func(int, []float64) error, error) {
	if len(numVpI) != 0 || len(inputDims) != 0 || len(inVals) != 0 || len(weights) != 0 {
		return nil, errors.Errorf("type Blank has no input deltas func for segments with inputs. Blank should be used exclusively for network inputs. (numVpI: %v, inputDims: %v, inVals: %v, weights: %v)", numVpI, inputDims, inVals, weights)
	}

	return func(a int, b []float64) error {return nil}, nil
}
func (t blank) CalculateFunc(inVals []float64, weights []float64, inputDims [][]int, dims []int, values []float64) (func() error, error) {
	if len(inVals) != 0 || len(weights) != 0 || len(inputDims) != 0 {
		return nil, errors.Errorf("type Blank has no calculate func for segments with inputs. Blank should be used exclusively for network inputs. (inVals: %v, weights: %v, inputDims: %v)", inVals, weights, inputDims)
	}

	return func() error {return nil}, nil
}
func (t blank) AdjustFunc(deltas []float64, weights []float64, inVals []float64) (func(float64) error, error) {
	if len(weights) != 0 || len(inVals) != 0 {
		return nil, errors.Errorf("type Blank has no adjust func for segments with inputs. Blank should be used exclusively for network inputs. (weights: %v, inVals: %v)", weights, inVals)
	}

	return func(a float64) error {return nil}, nil
}

// Softmax exponentially scales all of its inputs so that they add up to 1
type softmax int8

func Softmax() softmax {
	return softmax(0)
}

func (t softmax) NumValuesAndWeights(inputDims [][]int, numVpI []int, numInputs int, dims []int) (int, int, error) {
	if len(inputDims) <= 0 || len(numVpI) <= 0 || numInputs <= 0 {
		return 0, 0, errors.Errorf("can't give num values/weights: Softmax requires inputs. (inputDims: %v, numVpI: %v, numInputs: %v)", inputDims, numVpI, numInputs)
	}

	return numInputs, 0, nil
}
func (t softmax) InitWeights(weights []float64, inputDims [][]int, numVpI []int, numInputs int, dims []int) error {
	if len(weights) != 0 {
		return errors.Errorf("can't initialize weights: Softmax should not have weights. (weights: %v)", weights)
	} else if len(inputDims) <= 0 || len(numVpI) <= 0 || numInputs <= 0 {
		return errors.Errorf("can't initialize weights: Softmax should have inputs. (inputDims: %v, numVpI: %v, numInputs: %v)", inputDims, numVpI, numInputs)
	}

	return nil
}
func (t softmax) InputDeltasFunc(numVpI []int, inputDims [][]int, inVals, weights, values, deltas []float64) (func(int, []float64) error, error) {
	numInputs := 0
	for _, nv := range numVpI {
		numInputs += nv
	}
	if numInputs != len(inVals) {
		return nil, errors.Errorf("can't give input deltas func: total number of inputs != len(inVals) (%d != %d) (inVals: %v)", numInputs, len(inVals), inVals)
	}


	return func(ind int, ds []float64) error {
		start, err := deltasHelper(ind, numVpI, inVals)
		if err != nil {
			return err
		}
		for i := range ds {
			v := values[start + i]
			ds[i] = v * (1 - v) * deltas[start + i]
		}

		return nil
	}, nil
}
func (t softmax) CalculateFunc(inVals []float64, weights []float64, inputDims [][]int, dims []int, values []float64) (func() error, error) {
	if len(inVals) != len(values) {
		return nil, errors.Errorf("can't make calculate(), len(inVals) != len(values) (%d != %d)", len(inVals), len(values))
	}

	// @OPTIMIZE : any CalculateFunc is a good opportunity to make the program run faster (probably through porting it to C)
	return func() error {
		sum := 0.0
		for i, iv := range inVals {
			values[i] = math.Exp(iv)
			sum += values[i]
		}
		for i := range values {
			values[i] /= sum
		}
		return nil
	}, nil
}
func (t softmax) AdjustFunc(deltas []float64, weights []float64, inVals []float64) (func(float64) error, error) {
	if len(weights) != 0 {
		return nil, errors.Errorf("Softmax should have no weights. (weights: %v)", weights)
	}

	return func(learningRate float64) error {
		return nil
	}, nil
}

// Tanh feeds all of its inputs through the Tanh function (min: -1, max: 1)
type tanh int8

func Tanh() tanh {
	return tanh(0)
}

func (t tanh) NumValuesAndWeights(inputDims [][]int, numVpI []int, numInputs int, dims []int) (int, int, error) {
	if len(inputDims) <= 0 || len(numVpI) <= 0 || numInputs <= 0 {
		return 0, 0, errors.Errorf("can't give num values/weights: Tanh requires inputs. (inputDims: %v, numVpI: %v, numInputs: %v)", inputDims, numVpI, numInputs)
	}

	return numInputs, 0, nil
}
func (t tanh) InitWeights(weights []float64, inputDims [][]int, numVpI []int, numInputs int, dims []int) (error) {
	if len(weights) != 0 {
		return errors.Errorf("can't initialize weights: Tanh should not have weights. (weights: %v)", weights)
	} else if len(inputDims) <= 0 || len(numVpI) <= 0 || numInputs <= 0 {
		return errors.Errorf("can't initialize weights: Tanh should have inputs. (inputDims: %v, numVpI: %v, numInputs: %v)", inputDims, numVpI, numInputs)
	}

	return nil
}
func (t tanh) InputDeltasFunc(numVpI []int, inputDims [][]int, inVals []float64, weights []float64, values []float64, deltas []float64) (func(int, []float64) error, error) {
	numInputs := 0
	for _, nv := range numVpI {
		numInputs += nv
	}
	if numInputs != len(inVals) {
		return nil, errors.Errorf("can't give input deltas func: total number of inputs != len(inVals) (%d != %d) (inVals: %v)", numInputs, len(inVals), inVals)
	}

	return func(ind int, ds []float64) error {
		start, err := deltasHelper(ind, numVpI, inVals)
		if err != nil {
			return err
		}
		for i := range ds {
			ds[i] = (1 - math.Pow(values[start + i], 2)) * deltas[start + i]
		}

		return nil
	}, nil
}
func (t tanh) CalculateFunc(inVals []float64, weights []float64, inputDims [][]int, dims []int, values []float64) (func() error, error) {
	if len(inVals) != len(values) {
		return nil, errors.Errorf("can't make calculate(), len(inVals) != len(values) (%d != %d)", len(inVals), len(values))
	}

	// @OPTIMIZE : any CalculateFunc is a good opportunity to make the program run faster (probably through porting it to C)
	return func() error {
		for i, iv := range inVals {
			values[i] = math.Tanh(iv)
		}
		return nil
	}, nil
}
func (t tanh) AdjustFunc(deltas []float64, weights []float64, inVals []float64) (func(float64) error, error) {
	if len(weights) != 0 {
		return nil, errors.Errorf("Tanh should have no weights. (weights: %v)", weights)
	}

	return func(learningRate float64) error {
		return nil
	}, nil
}

// Neurons (just like a regular feed-forward deep network)
type neurons int8

func Neurons() neurons {
	return neurons(0)
}

func (t neurons) NumValuesAndWeights(inputDims [][]int, numVpI []int, numInputs int, dims []int) (int, int, error) {
	if len(inputDims) <= 0 || len(numVpI) <= 0 || numInputs <= 0 {
		return 0, 0, errors.Errorf("can't give num values/weights: Neurons requires inputs. (inputDims: %v, numVpI: %v, numInputs: %v)", inputDims, numVpI, numInputs)
	} else if dims[0] <= 0 {
		return 0, 0, errors.Errorf("can't give num values/weights: dims[0] must be >= 1 (dims: %v)", dims)
	}

	return dims[0], numInputs * dims[0], nil
}
func (t neurons) InitWeights(weights []float64, inputDims [][]int, numVpI []int, numInputs int, dims []int) (error) {
	if numInputs * dims[0] != len(weights) {
		return errors.Errorf("can't initialize weights: numInputs * dims[0] != len(weights) (%d * %d != %d)", numInputs, dims[0], len(weights))
	}

	for i := range weights {
		weights[i] = 1 / float64(dims[0]) * (2*rand.Float64() -1)
	}

	return nil
}
func (t neurons) InputDeltasFunc(numVpI []int, inputDims [][]int, inVals, weights, values, deltas []float64) (func(int, []float64) error, error) {
	numInputs := 0
	for _, nv := range numVpI {
		numInputs += nv
	}
	if numInputs != len(inVals) {
		return nil, errors.Errorf("can't give input deltas func: total number of inputs != len(inVals) (%d != %d) (inVals: %v)", numInputs, len(inVals), inVals)
	}

	return func(ind int, ds []float64) error {
		start, err := deltasHelper(ind, numVpI, inVals)
		if err != nil {
			return err
		}

		// @OPTIMIZE : this is a good place to start for optimizing things (port to C for matrix multiplication)
		for d := range ds {
			sum := 0.0
			for i :=  range deltas {
				sum += weights[i * len(inVals) + d + start] * deltas[i]
			}

			ds[d] = sum
		}

		return nil
	}, nil
}
func (t neurons) CalculateFunc(inVals []float64, weights []float64, inputDims [][]int, dims []int, values []float64) (func() error, error) {
	// @OPTIMIZE : can be made waaaay faster by porting to GPU-enabled matrix multiplication in C
	// weights are organized so that all of the weights for the first value are neighbors, then all for the second value, and so forth
	return func() error {
		i := 0
		for ind := range values {
			var v float64
			for _, iv := range inVals {
				v += iv * weights[i]
				i++
			}

			values[ind] = v
		}
		return nil
	}, nil
}
func (t neurons) AdjustFunc(deltas []float64, weights []float64, inVals []float64) (func(float64) error, error) {
	return func(learningRate float64) error {
		for v := range deltas {
			for i := range inVals {
				weights[v*len(inVals) + i] += -1 * learningRate * inVals[i] * deltas[v]
			}
		}
		return nil
	}, nil
}
