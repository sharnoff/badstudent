package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

// LSTM creates a standard LSTM unit, given the Nodes to input to the forget,
// ignore, and select gates, in addition to the input to the module -- 'update'.
//
// Returns the output of the select gate, without delay, so it can be used as
// Network outputs, in addition to the cell-state and a connection with a delay of
// 1. Related Note: Because there is no delay in the returned outputs, any
// implementation that uses this must add in their own delay in order to feed into
// another part of the network.
//
// To add peep-holes, simply use cell and cellDelay as inputs to the rest of the
// network where fitting.
func LSTM(net *bs.Network, size int, namePrefix string, update, ignore, forget, sel *bs.Node) (outputs, cell, cellDelay *bs.Node) {
	// for more information, consult: https://www.youtube.com/watch?v=WCUNPb-5EYI, at 20:31

	p := namePrefix

	// csLoop is the cell-state, with delay 1
	csLoop := net.Placeholder(p+"cell-state loop", size)

	// cs is the cell-state
	cs := net.Add(p+"cell-state", Add(), size,
		net.Add(p+"forget gate", Mult(), size, csLoop, forget),
		net.Add(p+"ignore gate", Mult(), size, update, ignore))

	csLoop.Replace(Identity(), cs).SetDelay(1)

	cst := net.Add(p+"cell-state tanh", Tanh(), size, cs)
	sGate := net.Add(p+"select gate", Mult(), size, cst, sel)

	return sGate, cs, csLoop
}

// ****************************************
// Add
// ****************************************

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

// ****************************************
// Mult
// ****************************************

type mult int8

// Mult returns an elementwise multiplication operator that implements
// badstudent.Operator. All inputs to its Node must have size equal to the Node.
func Mult() mult {
	return mult(0)
}

func (t mult) TypeString() string {
	return "multiply"
}

func (t mult) Finalize(n *bs.Node) error {
	for i := 0; i < n.NumInputNodes(); i++ {
		if n.InputSize(i) != n.Size() {
			return errors.Errorf("All inputs must have size equal to node (n.InputSize(%d) (%d) != %d)", i, n.InputSize(i), n.Size())
		}
	}

	return nil
}

func (t mult) Evaluate(n *bs.Node, values []float64) {
	inputs := n.CopyOfInputs()

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
