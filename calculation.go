package badstudent

import (
	"github.com/sharnoff/badstudent/utils"
	"fmt"
)

type status int8

const (
	initialized status = iota // 0
	finalized   status = iota // 1
	evaluated   status = iota // 2
	deltas      status = iota // 3
	adjusted    status = iota // 4
)

// Sets all Nodes' field 'completed' to false
func (net *Network) resetCompletion() {
	for _, n := range net.nodesByID {
		n.completed = false
	}
}

// ClearDelays clears all delay-related functions for all nodes in the Network. It
// essentially flushes a recurrent-type Network.
func (net *Network) ClearDelays() {
	for _, n := range net.nodesByID {
		for i := 0; i < cap(n.delay); i++ {
			<-n.delay
			n.delay <- make([]float64, n.values.Size())

			<-n.delayDeltas
			n.delayDeltas <- make([]float64, n.values.Size())
		}

		n.storedValues = nil
	}
}

// DoesNotAffectOutputsError results from one (or more) Node not having an output path to the
// Network outputs.
type DoesNotAffectOutputsError struct {
	N *Node
}

func (err DoesNotAffectOutputsError) Error() string {
	return fmt.Sprintf("Node %v does not affect outputs", err.N)
}

// InstantCycleError is a result of a zero-delay cycle in a network, i.e. at least one Node
// recieves input from itself with no delay.
type InstantCycleError struct {
	// Stack is the list of Nodes that form the cycle, with the first Node reported at both the
	// start and the end.
	Stack []*Node
}

func (err InstantCycleError) Error() string {
	list := "[" + err.Stack[len(err.Stack)-1].String() + "]"
	for i := len(err.Stack) - 2; i >= 0; i-- {
		list += " -> [" + err.Stack[i].String() + "]"
	}

	return "Network contains zero-delay cycle: " + list
}

// Checks:
// * all nodes affect outputs
// * there are no loops with zero delay
// Determines:
// * each node's need for calculating input deltas
//
// returns DoesNotAffectOutputsError or InstantCycleError
func (net *Network) checkGraph() error {
	if net.stat >= finalized {
		return nil
	}

	// Check all nodes affect outputs
	{
		var mark func(*Node)
		mark = func(n *Node) {
			if n.completed {
				return
			}

			n.completed = true

			if !n.IsInput() {
				for _, in := range n.inputs.nodes {
					mark(in)
				}
			}
			
			return
		}

		// Mark all Nodes that affect the network outputs.
		for _, out := range net.outputs.nodes {
			mark(out)
		}

		// If any Nodes don't affect outputs, return error
		for _, n := range net.nodesByID {
			if n.completed == false {
				return DoesNotAffectOutputsError{n}
			}
		}

		net.resetCompletion()
	}

	// Check that there are no loops with zero delay, from each Node to itself
	if net.mayHaveLoop {
		var check func(*Node) bool

		for _, root := range net.nodesByID {
			if root.IsInput() || num(root.inputs) == 0 {
				// this node can't recieve itself as input (directly or indirectly)
				// if it has no inputs
				continue
			} else if num(root.outputs) == 0 {
				// this node can't output to itself if it has no outputs
				continue
			} else if root.HasDelay() {
				// This node outputs to everything with delay, so it cannot
				// output to itself with none
				continue
			}

			// only used when check returns false
			var stack []*Node

			check = func(n *Node) bool {
				if n.completed {
					return true
				} else if n.HasDelay() {
					// cannot be part of a 0-delay loop
					return true
				}

				if n == root {
					stack = []*Node{n}
					return false
				} else {
					n.completed = true

					for _, out := range n.outputs.nodes {
						if !check(out) {
							stack = append(stack, n)
							return false
						}
					}

					return true
				}
			}

			for _, out := range root.outputs.nodes {
				if !check(out) {
					return InstantCycleError{stack}
				}
			}

			net.resetCompletion()
		}
	}

	// determine each node's need for calculating input deltas
	{
		// if deltas should not be calculated, it will be indicated by the
		// deltas of the Node having length 0

		var matter func(*Node, bool)
		matter = func(n *Node, dm bool) { // 'dm' is short for 'deltas matter'
			if n.completed && (len(n.deltas) != 0 || !dm) {
				return
			}

			if n.IsInput() {
				dm = false
			} else {
				dm = dm || n.adj != nil // if n is adjustable, deltas matter
			}

			if dm {
				n.deltas = make([]float64, n.Size())
			}
			n.completed = true

			for _, out := range n.outputs.nodes {
				matter(out, dm)
			}

			return
		}

		for _, in := range net.inputs.nodes {
			matter(in, false)
		}

		for _, n := range net.nodesByID {
			n.calcInDeltas = false

			if len(n.deltas) == 0 {
				continue
			}

			for _, in := range n.inputs.nodes {
				if len(in.deltas) != 0 {
					n.calcInDeltas = true
					break
				}
			}
		}

		net.resetCompletion()
	}

	return nil
}

// setValues does not check length
func (n *Node) setValues(values []float64) {
	if n.group == nil {
		n.values.Values = values
	} else {
		copy(n.values.Values, values)
	}
}

// These are completely arbitrary.
const (
	threadsPerCPU int = 2
	opsPerThread  int = 10
)

// Assumes n is NOT an input Node
func (n *Node) calculateValues() {
	if n.lyr != nil {
		n.lyr.Evaluate(n, n.values.Values)
		return
	}

	// we can assume n.elem != nil because the Operator would otherwise not be valid
	inputs := n.AllInputs()

	f := func(i int) {
		n.values.Values[i] = n.elem.Value(inputs[i], i)
	}

	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)
}

// isGettingDeltas is used for determining, during evaluation, whether or not push or pull from
// delay.
func (net *Network) isGettingDeltas() bool {
	return net.stat >= evaluated
}

// This function is a helper for adding clarity to *Node.evaluate()
func (n *Node) storeValues() {
	vs := make([]float64, n.Size())
	copy(vs, n.values.Values)
	n.storedValues = append(n.storedValues, vs)
}

// This function is a helper for adding clarity to *Node.evaluate()
func (n *Node) setFromStored() {
	n.setValues(n.storedValues[len(n.storedValues)-1])
	n.storedValues = n.storedValues[:len(n.storedValues)-1]
}

func (n *Node) evaluate() {
	if n.completed {
		return
	} else if n.IsInput() {
		if n.host.hasDelay {
			if n.host.isGettingDeltas() {
				n.setFromStored()
			} else {
				n.storeValues()
			}
		}

		n.completed = true
		return
	}

	if n.HasDelay() {
		if !n.host.isGettingDeltas() {
			// forward calculation, set from delay
			n.setValues(<-n.delay)
		} else {
			// backprop, set from stored
			n.setFromStored()
		}
		n.completed = true
	}

	for _, in := range n.inputs.nodes {
		in.evaluate()
	}

	if n.HasDelay() {
		n.values.Values = make([]float64, n.values.Size())
	}

	n.calculateValues()

	if n.HasDelay() && !n.host.isGettingDeltas() {
		n.delay <- n.values.Values
		n.storedValues = append(n.storedValues, n.values.Values)
	}

	// if we're backpropagating, don't worry about setting values; they were
	// set earlier

	n.completed = true
}

// evaluate updates the values of every Node to reflect the inputs. If the Network is not recurrent
// and has already been evaluated, it will do nothing.
//
// evaluate will return ErrNetNotFinalized if the Network has not been finalized.
func (net *Network) evaluate() error {
	if net.stat < finalized {
		return ErrNetNotFinalized
	} else if !net.hasDelay && net.stat >= evaluated {
		return nil
	}

	for _, out := range net.outputs.nodes {
		out.evaluate()
	}

	net.resetCompletion()
	net.stat = evaluated

	return nil
}

// calculateInputDeltas is a helper function that does what it says.
func (n *Node) calculateInputDeltas() {
	if n.lyr != nil {
		ds := n.lyr.InputDeltas(n)
		n.inputs.addDeltas(ds)

		return
	}

	start := 0
	for _, in := range n.inputs.nodes {
		end := start + in.Size()

		// If a Node does not need its deltas calculated, len(deltas) will be equal to zero.
		if len(in.deltas) == 0 {
			start = end
			continue
		}

		var ds []float64
		if in.HasDelay() {
			ds = in.tempDelayDeltas
		} else {
			ds = in.deltas
		}

		f := func(x int) {
			ds[x-start] += n.elem.Deriv(n, x) * n.deltas[x]
		}

		utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

		start = end
	}
}

// inputDeltas is only called by (*Network).getDeltas(). This will recurse upwards (towards
// outputs) through the Network.
func (n *Node) inputDeltas() {
	if n.completed {
		return
	}

	// For Nodes with delay, we need to calculate their input deltas first in order to avoid
	// recursive loops. We assume that nodes with delay have had their various deltas prepared
	// already.
	// Nodes without delay are simple; we get their outputs to calculate their deltas, and then
	// they [the root node] calculate their own.

	if n.HasDelay() {
		if n.calcInDeltas {
			n.calculateInputDeltas()
		}

		n.completed = true
	}

	for _, o := range n.outputs.nodes {
		o.inputDeltas()
	}

	if !n.HasDelay() && n.calcInDeltas {
		n.calculateInputDeltas()
	}

	n.completed = true
}

// assumes len(targets) == net.OutputSize(), net.stat >= evaluated
func (net *Network) getDeltas(targets []float64) {
	// reset deltas. For nodes without a need to calculate deltas, this will keep len(deltas) = 0.
	for _, n := range net.nodesByID {
		if n.HasDelay() {
			n.tempDelayDeltas = make([]float64, len(n.deltas))
			n.deltas = <-n.delayDeltas
		} else {
			n.deltas = make([]float64, len(n.deltas))
		}
	}

	// add to output deltas
	if len(targets) != 0 {
		// we check if len(targets) is zero because recurrent models can exempt
		// certain outputs from having significance by indicating providing no
		// targets

		// Indicating 'false' for duplicating opens the possibility of cost
		// functions to corrupt data. This issue is not significant.
		ds := net.cf.Derivs(net.outputs.getValues(false), targets)
		net.outputs.addDeltas(ds)
	}

	// recurse through network
	for _, in := range net.inputs.nodes {
		in.inputDeltas()
	}

	// put temporary delay deltas back into delay
	for _, n := range net.nodesByID {
		if n.HasDelay() {
			n.delayDeltas <- n.tempDelayDeltas
		}
	}

	net.stat = deltas
	net.resetCompletion()
	return
}

// Assumptions:
//	* net.stat >= finalized
//	* len(targets[n]) == net.OutputSize(), for all n in range len(targets)
func (net *Network) adjustRecurrent(targets [][]float64, saveChanges bool) {
	for i := len(targets) - 1; i >= 0; i-- {
		net.evaluate()

		net.getDeltas(targets[i])

		// we use saveChanges=true here to prevent issues with
		net.adjust(true)
	}

	if !saveChanges {
		net.AddWeights()
	}
}

// does not use completion, as it iterates through every Node directly.
func (net *Network) adjust(saveChanges bool) {
	for _, n := range net.nodesByID {
		n.adjust(saveChanges)
	}

	if saveChanges {
		net.hasSavedChanges = true
	}

	return
}

func (n *Node) adjust(saveChanges bool) {
	if n.adj == nil {
		return
	}

	w := n.delayedWeights
	if !saveChanges {
		w = n.adj.Weights()
	} else if len(n.delayedWeights) == 0 {
		n.delayedWeights = make([]float64, len(n.adj.Weights()))
		w = n.delayedWeights
	}

	var adj Adjustable
	if n.pen != nil {
		adj = penAdj{n.adj}
	} else {
		adj = n.adj
	}

	n.opt.Run(n, adj, w)
}

// penAdj is a wrapper for the usual Adjustable found in Nodes, to allow for the same types of
// interaction, but with added penalties for each weight.
type penAdj struct {
	Adjustable
}

func (p penAdj) Grad(n *Node, index int) float64 {
	return n.pen.Penalize(n, n.adj, index)
}

func (p penAdj) Weights() []float64 {
	return p.Weights()
}

func (n *Node) addWeights() {
	if n.adj == nil || len(n.delayedWeights) == 0 {
		return
	}

	// essentially:
	/* for i := range n.delayedWeights {
		n.adj.Weights()[i] += n.delayedWeights[i]
	} */

	ws := n.adj.Weights()

	f := func(i int) {
		ws[i] += n.delayedWeights[i]
	}

	utils.MultiThread(0, len(ws), f, opsPerThread, threadsPerCPU)
	n.delayedWeights = make([]float64, len(ws))
}

// Updates the weights in the network with any previously saved changes.
// Only runs if there are changes that have not been applied
func (net *Network) AddWeights() {
	if !net.hasSavedChanges {
		return
	}

	for _, n := range net.nodesByID {
		n.addWeights()
	}

	net.hasSavedChanges = false
	net.stat = finalized
	return
}
