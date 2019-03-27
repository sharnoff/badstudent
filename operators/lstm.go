package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"github.com/sharnoff/tensors"
	"runtime"
)

// LSTM creates a standard LSTM unit, given the Nodes to input to the forget, ignore, and select
// gates, in addition to the input to the module -- 'update'.
//
// Returns the output of the select gate, without delay, so it can be used as Network outputs, in
// addition to the cell-state and a connection with a delay of 1. Related Note: Because there is no
// delay in the returned outputs, any implementation that uses this must add in their own delay in
// order to feed into another part of the network.
//
// To add peep-holes, simply use cell and cellDelay as inputs to the rest of the network where
// fitting.
func LSTM(net *bs.Network, size int, update, ignore, forget, sel *bs.Node) (outputs, cell, cellDelay *bs.Node) {
	// for more information, consult: https://www.youtube.com/watch?v=WCUNPb-5EYI, at 20:31

	csLoop := net.Placeholder([]int{size})   // cell-state loop
	fGate := net.Add(Mult(), csLoop, forget) // forget gate
	iGate := net.Add(Mult(), update, ignore) // ignore gate
	cs := net.Add(Add(), fGate, iGate)       // cell-state
	csLoop.Replace(Identity(), cs).SetDelay(1)

	cst := net.Add(Tanh(), cs)         // cell-state tanh
	sGate := net.Add(Mult(), cst, sel) // select gate

	return sGate, cs, csLoop
}

// ****************************************
// Add
// ****************************************

type add int8

// Add returns an elementwise addition operator that implements badstudent.Operator. All inputs to
// its Node must have size equal to the Node.
func Add() add {
	return add(0)
}

func (t add) TypeString() string {
	return "add"
}

func (t add) Finalize(n *bs.Node) error {
	s := n.Input(0).Size()
	for i := 1; i < n.NumInputNodes(); i++ {
		if n.Input(i).Size() != s {
			return errors.Errorf("All inputs must have equal size (n.InputSize(%d) (%d) != n.InputSize(%d) (%d))",
				i, n.Input(i).Size(), 0, n.Input(0).Size())
		}
	}

	return nil
}

func (t add) OutputShape(inputs []*bs.Node) (tensors.Tensor, error) {
	return tensors.NewTensor(inputs[0].Dims()), nil
}

func (t add) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()

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

// ****************************************
// Mult
// ****************************************

type mult int8

// Mult returns an elementwise multiplication operator that implements badstudent.Operator. All
// inputs to its Node must have size equal to the Node.
func Mult() mult {
	return mult(0)
}

func (t mult) TypeString() string {
	return "multiply"
}

func (t mult) Finalize(n *bs.Node) error {
	s := n.Input(0).Size()
	for i := 1; i < n.NumInputNodes(); i++ {
		if n.Input(i).Size() != s {
			return errors.Errorf("All inputs must have equal size (n.InputSize(%d) (%d) != n.InputSize(%d) (%d))",
				i, n.Input(i).Size(), 0, n.Input(0).Size())
		}
	}

	return nil
}

func (t mult) OutputShape(inputs []*bs.Node) (tensors.Tensor, error) {
	return tensors.NewTensor(inputs[0].Dims()), nil
}

func (t mult) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()

	f := func(i int) {
		values[i] = inputs[i]
		for in := 1; in < n.NumInputNodes(); in++ {
			values[i] *= inputs[in*n.Size()+i]
		}
	}

	// just random constants. Have not been optimized
	opsPerThread, threadsPerCPU := runtime.NumCPU()*2, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t mult) InputDeltas(n *bs.Node) []float64 {
	ds := make([]float64, n.NumInputs())

	f := func(i int) {
		ds[i] = n.Delta(i%n.Size()) * n.Value(i%n.Size()) / n.InputValue(i)
	}

	// just random constants. Have not been optimized
	opsPerThread, threadsPerCPU := runtime.NumCPU()*2, 1
	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)

	return ds
}
