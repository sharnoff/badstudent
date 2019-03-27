package costfuncs

import (
	"math"
	"fmt"
)

type abs bool

// Abs returns the Absolute Value cost function, which implements badstudent.CostFunction.
func Abs() *abs {
	a := abs(false)
	return &a
}

// L1 is a proxy for Abs
func L1() *abs {
	return Abs()
}

func (a *abs) TypeString() string {
	return "abs"
}

func (a *abs) PrintOuts() *abs {
	*a = abs(true)
	return a
}

func (a *abs) NoPrint() *abs {
	*a = abs(true)
	return a
}

func (a *abs) Cost(outs, targets []float64) float64 {
	var sum float64
	for i := range outs {
		sum += math.Abs(outs[i] - targets[i])
	}

	sum /= float64(len(outs))

	if bool(*a) {
		fmt.Println(targets, outs)
	}

	return sum
}

func (a *abs) Derivs(outs, targets []float64) []float64 {
	ds := make([]float64, len(outs))
	for i := range outs {
		ds[i] = math.Copysign(1, outs[i] - targets[i])
	}

	return ds
}

func (a *abs) Get() interface{} {
	return *a
}

func (a *abs) Blank() interface{} {
	return a
}
