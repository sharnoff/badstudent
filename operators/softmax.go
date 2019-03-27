package operators

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/tensors"
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
	return nil
}

func (t softmax) OutputShape(inputs []*bs.Node) (tensors.Tensor, error) {
	ls := make([]tensors.Tensor, len(inputs))
	for i := range ls {
		ls[i] = inputs[i].Shape()
	}

	return bs.ConcatShape(ls)
}

func (t softmax) Evaluate(n *bs.Node, values []float64) {
	// This could be multithreaded with forking.
	inputs := n.AllInputs()

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
