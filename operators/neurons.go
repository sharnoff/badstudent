package operators

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"github.com/sharnoff/tensors"
	"fmt"
)

type neurons struct {
	Size int

	// weights is organized in (n = number of values) sets of:
	// input, input, ... input, bias
	//
	// cannot be called 'Weights' because it needs method Weights
	Ws []float64

	// always either 0 or 1. It is represented as an integer to make the math easier and to reduce
	// the number of necessary conditionals
	NumBiases int

	// the value multiplied by bias
	Bias float64
}

// this is really either zero or 1
const default_numBiases int = 1

// Neurons returns a basic layer of perceptrons with biases that implements badstudent.Operator.
//
// The value of the biases can be set by BiasValue, and the number of biases can be set by Biases.
func Neurons(size int) *neurons {
	n := new(neurons)
	n.Size = size
	n.Bias = defaultValue["neurons-bias"]
	n.NumBiases = default_numBiases
	return n
}

// Dense is a substitute for Neurons
func Dense(size int) *neurons {
	return Neurons(size)
}

// FullyConnected is a substitute for Neurons
func FullyConnected(size int) *neurons {
	return Neurons(size)
}

// ***************************************************
// Customization Functions
// ***************************************************

// NoBiases changes the neurons to not have any biases.
func (n *neurons) NoBiases() *neurons {
	n.NumBiases = 0
	return n
}

// WithBiases changes the neurons to have biases, if they did not already
func (n *neurons) WithBiases() *neurons {
	n.NumBiases = 1
	return n
}

// BiasValue sets the value multiplied by the biases. The default value can be set by
// SetDefault("neurons-bias")
func (n *neurons) BiasValue(b float64) *neurons {
	n.Bias = b
	return n
}

// ***************************************************
// Helper Functions
// ***************************************************

// this makes it a little harder to optimize, but it'll all be done with matrices eventually, so it
// doesn't really matter
func (t *neurons) weight(n *bs.Node, in, val int) float64 {
	i := val*(n.NumInputs()+t.NumBiases)+in

	if i >= len(t.Ws) {
		fmt.Printf("len(t.Ws)=%d, val=%d, n.NumInputs()=%d, t.NumBiases=%d, in=%d\n",
				len(t.Ws), val, n.NumInputs(), t.NumBiases, in)
	}

	return t.Ws[val*(n.NumInputs()+t.NumBiases)+in]
}

// ***************************************************
// Interface-required functions
// ***************************************************

func (t *neurons) TypeString() string {
	return "neurons"
}

func (t *neurons) Finalize(n *bs.Node) error {
	// if it's been loaded from a file...
	if len(t.Ws) != 0 {
		return nil
	}

	t.Ws = make([]float64, (n.NumInputs()+t.NumBiases)*t.Size)
	return nil
}

func (t *neurons) Get() interface{} {
	return *t
}

func (t *neurons) Blank() interface{} {
	return t
}

func (t *neurons) OutputShape(inputs []*bs.Node) (tensors.Tensor, error) {
	return tensors.NewTensor([]int{t.Size}), nil
}

func (t *neurons) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()
	f := func(v int) {
		var sum float64
		for in := range inputs {
			sum += t.weight(n, in, v) * inputs[in]
		}

		if t.NumBiases != 0 {
			sum += t.Bias * t.weight(n, n.NumInputs(), v)
		}

		values[v] = sum
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t *neurons) InputDeltas(n *bs.Node) []float64 {
	ds := make([]float64, n.NumInputs())

	f := func(in int) {
		for v := 0; v < n.Size(); v++ {
			ds[in] += n.Delta(v) * t.weight(n, in, v)
		}
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, n.NumInputs(), f, opsPerThread, threadsPerCPU)

	return ds
}

func (t *neurons) Grad(n *bs.Node, index int) float64 {
	in := index % (n.NumInputs() + t.NumBiases)
	v := (index - in) / (n.NumInputs() + t.NumBiases)
	if in < n.NumInputs() {
		return n.InputValue(in) * n.Delta(v)
	} else {
		return t.Bias * n.Delta(v)
	}
}

func (t *neurons) Weights() []float64 {
	return t.Ws
}
