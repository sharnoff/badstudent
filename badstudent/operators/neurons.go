package operators

import (
	"github.com/sharnoff/smartlearning/badstudent"
	"github.com/pkg/errors"
	"math/rand"
)

type neurons struct {
	weights [][]float64
	biases  []float64
}

func Neurons() *neurons {
	return new(neurons)
}

const bias_value float64 = 1

func (n *neurons) Init(l *badstudent.Layer) (error) {
	// if l.NumInputLayers() == 0 {
	// 	return 0, errors.Errorf("Can't Init() operator 'neurons', layer %v has no inputs", l)
	// }

	n.weights = make([][]float64, l.Size())
	n.biases = make([]float64, l.Size())

	for v := range n.weights {
		n.weights[v] = make([]float64, l.NumInputs())
		for i := range n.weights[v] {
			n.weights[v][i] = (2 * rand.Float64() - 1) / float64(l.NumInputs())
		}

		n.biases[v] = (2 * rand.Float64() - 1) / float64(l.NumInputs())
	}

	return nil
}

func (n *neurons) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()
	for v := range values {
		var sum float64
		for in := range inputs {
			sum += inputs[in] * n.weights[v][in]
		}
		sum += bias_value * n.biases[v]

		// temporary implementation of logistic function
		// values[v] = 0.5 + 0.5 * math.Tanh(0.5 * sum)
		values[v] = sum
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

		// temporary implementation of logistic function
		// sum *= l.InputValue(in) * (1 - l.InputValue(in))
		add(in - start, sum)
	}

	return nil
}

func (n *neurons) Adjust(l *badstudent.Layer, opt badstudent.Optimizer, learningRate float64) error {
	inputs := l.CopyOfInputs()

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

			n.weights[v][in] += addend
		}

		if err := opt.Run(l, len(inputs) * l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on weights failed\n", l)
		}
	}

	// now run on biases
	{
		grad := func(index int) float64 {
			return bias_value * l.Delta(index)
		}

		add := func(index int, addend float64) {
			n.biases[index] += addend
		}

		if err := opt.Run(l, l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on biases failed\n", l)
		}
	}

	return nil
}