package badstudent

import (
	"fmt"
	"github.com/pkg/errors"
	"math"
)

type squarederror bool

// The standard squared error function, except for that it returns the average
// SquaredError is a provided implementation of a CostFunction
//
// if 'debug' is true, this will, at each call to Cost(), println(values, targets)
func SquaredError(debug bool) squarederror {
	return squarederror(debug)
}

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
		returnFunc(i-start, outputs[i]-targets[i])
	}

	return nil
}
