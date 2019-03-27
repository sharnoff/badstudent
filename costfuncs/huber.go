package costfuncs

import (
	"math"
	"fmt"
)

type huber struct {
	δ float64
	print bool
}

// Huber returns the Huber Loss Function, which implements badstudent.CostFunction. δ controls the
// bounds of the transition between MSE and Absolute Value.
func Huber(δ float64) *huber {
	h := huber{δ: δ}
	return &h
}

func (h *huber) TypeString() string {
	return "huber"
}

func (h *huber) PrintOuts() *huber {
	h.print = true
	return h
}

func (h *huber) NoPrint() *huber {
	h.print = false
	return h
}

func (h *huber) Cost(outs, targets []float64) float64 {
	var sum float64
	for i := range outs {
		d := math.Abs(outs[i] - targets[i])
		if d <= h.δ {
			sum += 0.5*d*d // faster than math.Pow
		} else {
			sum += h.δ * d - 0.5*h.δ*h.δ // faster than math.Pow
		}
	}

	sum /= float64(len(outs))

	if h.print {
		fmt.Println(targets, outs)
	}

	return sum
}

func (h *huber) Derivs(outs, targets []float64) []float64 {
	ds := make([]float64, len(outs))
	for i := range outs {
		d := outs[i] - targets[i]
		if !(d < -h.δ || d > h.δ) { // d >= -h.δ && d <= h.δ 
			ds[i] = d
		} else {
			ds[i] = h.δ*math.Copysign(1, outs[i] - targets[i])
		}
	}

	return ds
}

func (h *huber) Get() interface{} {
	fmt.Println("Getting")
	return *h
}

func (h *huber) Blank() interface{} {
	fmt.Println("Blank")
	return h
}
