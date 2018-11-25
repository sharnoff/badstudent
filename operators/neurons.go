package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"

	"encoding/json"
	"os"
)

type neurons struct {
	// weights is organized in (n = number of values) sets of:
	// input, input, ... input, bias
	//
	// cannot be called 'Weights' because it needs method Weights
	Ws []float64

	// always either 0 or 1. It is represented as an integer to make the math easier
	// and to reduce the number of necessary conditionals
	NumBiases int

	// the value multiplied by bias
	Bias float64
}

// this is really either zero or 1
const default_numBiases int = 1

// Neurons returns a basic layer of perceptrons with biases that implements
// badstudent.Operator.
//
// The value of the biases can be set by BiasValue, and the number of biases can be
// set by Biases.
func Neurons() *neurons {
	n := new(neurons)
	n.Bias = defaultValue["neurons-bias"]
	n.NumBiases = default_numBiases
	return n
}

// Dense is a substitute for Neurons
func Dense() *neurons {
	return Neurons()
}

// FullyConnected is a substitute for Neurons
func FullyConnected() *neurons {
	return Neurons()
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

// BiasValue sets the value multiplied by the biases. The default value can be set
// by SetDefault("neurons-bias")
func (n *neurons) BiasValue(b float64) *neurons {
	n.Bias = b
	return n
}

// ***************************************************
// Helper Functions
// ***************************************************

// this makes it a little harder to optimize, but it'll all be done with matrices
// eventually, so it doesn't really matter
func (t *neurons) weight(n *bs.Node, in, val int) float64 {
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
	if l := len(t.Ws); l != 0 {
		if l != (n.NumInputs()+t.NumBiases)*n.Size() {
			return errors.Errorf("Loaded number of weights and i/o dimensions do not match")
		}

		return nil
	}

	t.Ws = make([]float64, (n.NumInputs()+t.NumBiases)*n.Size())
	return nil
}

// encodes via JSON into 'weights.txt'
func (t *neurons) Save(dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Failed to create save directory")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Failed to create file %q in %q", "weights.txt", dirPath)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(t); err != nil {
		return errors.Errorf("Failed to encode JSON to file %q in %q", "weights.txt", dirPath)
	}

	return nil
}

// decodes JSON from 'weights.txt'
func (t *neurons) Load(dirPath string) error {
	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Failed to open file %q in %q", "weights.txt", dirPath)
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(t); err != nil {
		return errors.Errorf("Failed to decode JSON from file %q in %q", "weights.txt", dirPath)
	}

	return nil
}

func (t *neurons) Evaluate(n *bs.Node, values []float64) {
	inputs := n.CopyOfInputs()
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
