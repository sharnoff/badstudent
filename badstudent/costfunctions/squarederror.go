package costfunctions

import (
	"fmt"
	"github.com/pkg/errors"
	"math"
)

type squarederror bool

func SquaredError(debug bool) squarederror {
	return squarederror(debug)
}

// returns error if len(values) != len(targets)
func (c squarederror) Cost(values, targets []float64) (float64, error) {
	if len(values) != len(targets) {
		return 0, errors.Errorf("Can't get Cost() of 'squared error', len(values) != len(targets) (%d != %d)", len(values), len(targets))
	}

	if bool(c) {
		fmt.Println(values, targets)
	}

	var totalErr float64
	for i := range values {
		totalErr += math.Pow(values[i]-targets[i], 2)
	}

	return totalErr / float64(len(values)), nil
}

func (c squarederror) Deriv(outputs, targets []float64, start, end int, returnFunc func(int, float64)) error {

	for i := start; i < end; i++ {
		deriv := outputs[i] - targets[i]

		// to handle tanh, will be removed
		// deriv *= outputs[i] * (1 - outputs[i])
		returnFunc(i-start, deriv)
	}

	return nil
}
