package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"math"
)

type softmax int8

// Softmax returns the softmax function as a badstudent.Operator.
func Softmax() softmax {
	return softmax(0)
}

func (t softmax) TypeString() string {
	return "softmax"
}

func (t softmax) Finalize(n *bs.Node) error {
	if n.NumInputs() != n.Size() {
		return errors.Errorf("Number of inputs not equal to size. (%d != %d)", n.NumInputs(), n.Size())
	}

	return nil
}

func (t softmax) Evaluate(n *bs.Node, values []float64) {
	// This could be multithreaded with forking.
	inputs := n.CopyOfInputs()

	var sum float64
	for i := range values {
		values[i] = math.Exp(inputs[i])
		sum += values[i]
	}

	for i := range values {
		values[i] /= sum
	}
}

func (t softmax) InputDeltas(n *bs.Node) []float64 {
	ds := make([]float64, n.Size())
	for i := range ds {
		ds[i] = n.Value(i) * (1 - n.Value(i))
	}

	return ds
}
