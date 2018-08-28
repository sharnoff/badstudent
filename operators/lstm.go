package operators

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/pkg/errors"
)

// LSTM creates a standard LSTM unit, given the Nodes to input to the
// forget, ignore, and select gates, in addition to the input to the
// module -- 'update'.
//
// Returns the output of the select gate, without delay, so it can be used
// as Network outputs, in addition to the cell-state and a connection with
// a delay of 1. Related Note: Because there is no delay in the returned
// outputs, any implementation that uses this must add in their own delay
// in order to feed into another part of the network.
//
// To add peep-holes, simply use cell and cellDelay as inputs to the rest
// of the network where fitting.
//
// Loading:
//
// For loading, the operators for each name are:
// * namePrefix + "cell-state loop": Identity()
// * namePrefix + "forget gate": Mult()
// * namePrefix + "ignore gate": Mult()
// * namePrefix + "cell-state": Add()
// * namePrefix + "cell-state tanh": Tanh()
//
// Structure:
//
// 'X' signifies an element-wise multiplication; Mult();
// '+' signifies an element-wise addition; Add();
// ->  signifies input (x -> y means x inputs to y);
// --> signifies input with delay;
//
// 'cell-state' --> 'cell state delay';
// 'cell-state delay' X 'forget' -> 'forget gate';
// 'ignore' X 'update' -> 'ignore gate';
// 'forget gate' + 'ignore gate' -> 'cell-state';
// tanh('cell-state') X 'sel' -> 'select gate';
// outputs = 'select gate'
//
// returning the cell-state allows for peep-holes
func LSTM(net *bs.Network, size int, namePrefix string, forget, update, ignore, sel *bs.Node) (outputs, cell, cellDelay *bs.Node, err error) {
	var sGate, cs, iGate, fGate, csLoop *bs.Node // csLoop is the cell-state delay

	csLoop, err = net.Placeholder(namePrefix+"cell-state loop", size)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add cell-state loop\n")
	}
	cellDelay = csLoop

	fGate, err = net.Add(namePrefix+"forget gate", Mult(), size, bs.NoDelay, csLoop, forget)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add forget gate\n")
	}

	iGate, err = net.Add(namePrefix+"ignore gate", Mult(), size, bs.NoDelay, update, ignore)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add ignore gate\n")
	}

	cs, err = net.Add(namePrefix+"cell-state", Add(), size, bs.NoDelay, fGate, iGate)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add cell-state\n")
	}
	cell = cs

	err = csLoop.Replace(Identity(), bs.Delay(1), cs)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to replace cell-state loop\n")
	}

	cs, err = net.Add(namePrefix+"cell-state tanh", Tanh(), size, bs.NoDelay, cs)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add cell-state activation (tanh)\n")
	}

	sGate, err = net.Add(namePrefix+"select gate", Mult(), size, bs.NoDelay, cs, sel)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to add select gate\n")
	}
}