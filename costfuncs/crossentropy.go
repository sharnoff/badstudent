package costfuncs

import (
	"math"
	"fmt"
)

type crossEntropy bool

func CrossEntropy() *crossEntropy {
	c := crossEntropy(false)
	return &c
}

func NegativeLog() *crossEntropy {
	return CrossEntropy()
}

func (c *crossEntropy) TypeString() string {
	return "cross-entropy"
}

func (c *crossEntropy) PrintOuts() *crossEntropy {
	*c = crossEntropy(true)
	return c
}

func (c *crossEntropy) NoPrint() *crossEntropy {
	*c = crossEntropy(false)
	return c
}

func (c *crossEntropy) Cost(outs, targets []float64) float64 {
	var sum float64
	for i := range outs {
		sum -= outs[i] * math.Log(targets[i])
	}

	sum /= float64(len(outs))

	if bool(*c) {
		fmt.Println(targets, outs)
	}

	return sum
}

func (c *crossEntropy) Derivs(outs, targets []float64) []float64 {
	ds := make([]float64, len(outs))
	for i := range outs {
		ds[i] = - targets[i] / outs[i]
	}
	
	return ds
}

func (c *crossEntropy) Get() interface{} {
	return *c
}

func (c *crossEntropy) Blank() interface{} {
	return c
}
