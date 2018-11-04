package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

type add int8

// Add returns an elementwise addition operator that implements badstudent.Operator.
// All inputs to its Node must have size equal to the Node.
func Add() add {
	return add(0)
}

func (t add) TypeString() string {
	return "add"
}

func (t add) Finalize(n *bs.Node) error {
	for i := 0; i < n.NumInputNodes(); i++ {
		if n.InputSize(i) != n.Size() {
			return errors.Errorf("All inputs must have size equal to node (n.InputSize(%d) (%d) != %d)", i, n.InputSize(i), n.Size())
		}
	}

	return nil
}

func (t add) Evaluate(n *bs.Node, values []float64) {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = inputs[i]
		for in := 1; in < n.NumInputNodes(); in++ {
			values[i] += inputs[in*n.Size()+i]
		}
	}

	// just random constants. Have not been optimized
	opsPerThread, threadsPerCPU := runtime.NumCPU()*2, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t add) InputDeltas(n *bs.Node) []float64 {
	ds := make([]float64, n.NumInputs())

	f := func(i int) {
		ds[i] = n.Delta(i % n.Size())
	}

	// just random constants. Have not been optimized
	opsPerThread, threadsPerCPU := runtime.NumCPU()*2, 1
	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)

	return ds
}
