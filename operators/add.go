package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

type add int8

// Adds its inputs. All inputs must have the same size as node.
// Can loop on itself to make LSTM
func Add() add {
	return add(0)
}

func (t add) Init(n *bs.Node) error {
	if n.NumInputs() == 0 {
		return errors.Errorf("Must have >= 1 input")
	}

	for i := 0; i < n.NumInputNodes(); i++ {
		if n.InputSize(i) != n.Size() {
			return errors.Errorf("Size of input %d is not equal to node (%d != %d)", i, n.InputSize(i), n.Size())
		}
	}

	return nil
}

// does not save anything
func (t add) Save(n *bs.Node) error {
	return nil
}

// does not save anything
func (t add) Load(n *bs.Node) error {
	return nil
}

func (t add) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = inputs[i]
		for in := 1; in < n.NumInputNodes(); in++ {
			values[i] += inputs[in*n.Size()+i]
		}
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t add) Value(n *bs.Node, index int) float64 {

	v := n.InputValue(index)
	for in := 1; in < n.NumInputNodes(); in++ {
		v += n.InputValue(in*n.Size() + index)
	}

	return v
}

func (t add) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {
	f := func(i int) {
		add(i-start, n.Delta(i%n.Size()))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t add) CanBeAdjusted(n *bs.Node) bool {
	return false
}

func (t add) NeedsValues(n *bs.Node) bool {
	return false
}

func (t add) NeedsInputs(n *bs.Node) bool {
	return false
}

func (t add) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t add) AddWeights(n *bs.Node) error {
	return nil
}
