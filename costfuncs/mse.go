package costfuncs

import (
	"math"
	"fmt"
)

type mse bool

// MSE returns the mean squared error cost function, which implements
// badstudent.CostFunction
func MSE() *mse {
	m := mse(false)
	return &m
}

func (m *mse) TypeString() string {
	return "mse"
}

func (m *mse) PrintOuts() *mse {
	*m = mse(true)
	return m
}

func (m *mse) NoPrint() *mse {
	*m = mse(false)
	return m
}

func (m *mse) Cost(outs, targets []float64) float64 {
	var sum float64
	for i := range outs {
		sum += math.Pow(outs[i]-targets[i], 2)
	}

	if bool(*m) {
		fmt.Println(targets, outs)
	}

	return sum / float64(len(outs))
}

func (m *mse) Derivs(outs, targets []float64) []float64 {
	ds := make([]float64, len(outs))
	for i := range outs {
		ds[i] = outs[i] - targets[i]
	}

	return ds
}
