package costfuncs

import (
	"math"
	"fmt"
)

type mse bool

// MSE returns the mean squared error cost function, which implements badstudent.CostFunction.
func MSE() *mse {
	m := mse(false)
	return &m
}

// L2 is a proxy for MSE
func L2() *mse {
	return MSE()
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
		sum += 0.5 * math.Pow(outs[i]-targets[i], 2)
	}

	sum /= float64(len(outs))

	if bool(*m) {
		fmt.Println(targets, outs)
	}

	return sum
}

func (m *mse) Derivs(outs, targets []float64) []float64 {
	ds := make([]float64, len(outs))
	for i := range outs {
		ds[i] = outs[i] - targets[i]
	}

	return ds
}

func (m *mse) Get() interface{} {
	return *m
}

func (m *mse) Blank() interface{} {
	return m
}
