package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/smartlearning/badstudent"
	"math/rand"
)

type neurons struct {
	weights [][]float64
	biases  []float64

	weightChanges [][]float64
	biasChanges   []float64
}

func Neurons() *neurons {
	return new(neurons)
}

const bias_value float64 = 1

func (n *neurons) Init(l *badstudent.Layer) error {

	n.weights = make([][]float64, l.Size())
	n.weightChanges = make([][]float64, l.Size())
	n.biases = make([]float64, l.Size())
	n.biasChanges = make([]float64, l.Size())

	for v := range n.weights {
		n.weights[v] = make([]float64, l.NumInputs())
		n.weightChanges[v] = make([]float64, l.NumInputs())
		for i := range n.weights[v] {
			n.weights[v][i] = (2*rand.Float64() - 1) / float64(l.NumInputs())
		}

		n.biases[v] = (2*rand.Float64() - 1) / float64(l.NumInputs())
	}

	return nil
}

func (n *neurons) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()
	for v := range values {
		var sum float64
		for in := range inputs {
			sum += n.weights[v][in] * inputs[in]
		}

		values[v] = sum + (n.biases[v] * bias_value)
		sum += bias_value * n.biases[v]
	}

	return nil
}

func (n *neurons) InputDeltas(l *badstudent.Layer, add func(int, float64), input int) error {
	start := l.PreviousInputs(input)
	end := start + l.InputSize(input)

	for in := start; in < end; in++ {
		var sum float64
		for v := 0; v < l.Size(); v++ {
			sum += l.Delta(v) * n.weights[v][in]
		}

		add(in-start, sum)
	}

	return nil
}

func (n *neurons) Adjust(l *badstudent.Layer, opt badstudent.Optimizer, learningRate float64, saveChanges bool) error {
	inputs := l.CopyOfInputs()

	targetWeights := n.weightChanges
	targetBiases := n.biasChanges
	if !saveChanges {
		targetWeights = n.weights
		targetBiases = n.biases
	}

	// first run on weights, then biases
	{
		grad := func(index int) float64 {
			in := index % len(inputs)
			v := (index - in) / len(inputs)

			return inputs[in] * l.Delta(v)
		}

		add := func(index int, addend float64) {
			in := index % len(inputs)
			v := (index - in) / len(inputs)

			targetWeights[v][in] += addend
		}

		if err := opt.Run(l, len(inputs)*l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on weights failed\n", l)
		}
	}

	// now run on biases
	{
		grad := func(index int) float64 {
			return bias_value * l.Delta(index)
		}

		add := func(index int, addend float64) {
			targetBiases[index] += addend
		}

		if err := opt.Run(l, l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on biases failed\n", l)
		}
	}

	return nil
}

func (n *neurons) AddWeights(l *badstudent.Layer) error {
	for v := range n.weights {
		for in := range n.weights[v] {
			n.weights[v][in] += n.weightChanges[v][in]
		}
		n.biases[v] += n.biasChanges[v]

		n.weightChanges[v] = make([]float64, len(n.weights[v]))
	}
	n.biasChanges = make([]float64, len(n.biases))

	return nil
}
