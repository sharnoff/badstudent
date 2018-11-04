package operators

import (
	bs "github.com/sharnoff/badstudent"
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
