package badstudent

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
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
			n.delay <- make([]float64, len(n.values))

			<-n.delayDeltas
			n.delayDeltas <- make([]float64, len(n.values))
		}

		n.storedValues = nil
	}
}

// Checks:
// * all nodes affect outputs
// * there are no loops with zero delay
// Determines:
// * each node's need for calculating input deltas
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

			for _, in := range n.inputs.nodes {
				mark(in)
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
				return errors.Errorf("Node %v does not affect Network outputs", n)
			}
		}

		net.resetCompletion()
	}

	// Check that there are no loops with zero delay, from each Node to itself
	if net.mayHaveLoop {
		var check func(*Node) error

		for _, root := range net.nodesByID {
			if num(root.inputs) == 0 {
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

			check = func(n *Node) error {
				if n.completed {
					return nil
				} else if n.HasDelay() {
					// cannot be part of a 0-delay loop
					return nil
				}

				if n == root {
					return errors.Errorf("Node %v receives input from itself with no delay", root)
				} else {
					n.completed = true

					for _, out := range n.outputs.nodes {
						if err := check(out); err != nil {
							return errors.Wrapf(err, "%v to", n)
						}
					}

					return nil
				}
			}

			for _, out := range root.outputs.nodes {
				if err := check(out); err != nil {
					return err
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

func (n *Node) setValues(values []float64) {
	if n.group == nil {
		n.values = values
	} else {
		copy(n.values, values)
	}
}

const (
	threadsPerCPU        int = 2
	threadSizeMultiplier int = 10
)

func (n *Node) calculateValues(vs []float64) {
	if n.lyr != nil {
		n.lyr.Evaluate(n, vs)
	} else { // n.el must not be nil in order for n.typ to be valid
		inputs := n.CopyOfInputs()

		f := func(i int) {
			n.values[i] = n.el.Value(inputs[i], i)
		}

		opsPerThread := runtime.NumCPU() * threadSizeMultiplier
		utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)
	}
}

// Note: this is only really applicable for input Nodes
func (n *Node) storeValues() {
	vs := make([]float64, n.Size())
	copy(vs, n.values)
	n.storedValues = append(n.storedValues, vs)
}

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

	values := n.values
	if n.HasDelay() {
		values = make([]float64, len(n.values))
	}

	n.calculateValues(values)

	if n.HasDelay() && !n.host.isGettingDeltas() {
		n.delay <- values
		n.storedValues = append(n.storedValues, values)
	}

	// if we're backpropagating, don't worry about setting values; they were
	// set earlier

	n.completed = true
}

// Changes the values of the Nodes so that they accurately reflect the inputs
//
// 'recurrent' forces ignoring avoiding repetition if net.stat >= evaluated, if true
//
// returns error iff the network has not been finalized
func (net *Network) evaluate(recurrent bool) error {
	if net.stat < finalized {
		return errors.Errorf("Network is not complete")
	} else if !recurrent && net.stat >= evaluated {
		return nil
	}

	for _, out := range net.outputs.nodes {
		out.evaluate()
	}

	net.resetCompletion()
	net.stat = evaluated

	return nil
}

func (n *Node) calculateInputDeltas() {
	if n.lyr != nil {
		ds := n.lyr.InputDeltas(n)
		n.inputs.addDeltas(ds)
	} else {
		// could be improved to only calculate for inputs that need deltas
		start := 0
		for i, in := range n.inputs.nodes {
			if len(in.deltas) == 0 {
				continue
			}

			end := start + n.inputs.nodes[i].Size()

			var ds []float64
			if in.HasDelay() {
				ds = in.tempDelayDeltas
			} else {
				ds = in.deltas
			}

			f := func(x int) {
				ds[x-start] += n.el.Deriv(n, x) * n.deltas[x]
			}

			opsPerThread := runtime.NumCPU() * threadSizeMultiplier
			utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

			start = end
		}
	}
}

// can only be called by (*Network).getDeltas()
// recurses towards outputs, even if calculating input deltas is unnecessary
func (n *Node) inputDeltas() {
	if n.completed {
		return
	} else if !n.calcInDeltas {
		// recurse towards outputs
		for _, o := range n.outputs.nodes {
			o.inputDeltas()
		}

		n.completed = true
	}

	if num(n.outputs) != 0 && n.Delay() == 0 {
		// make sure that this node's deltas are calculated.
		// if it's an output, they have already been added to by the network
		for _, o := range n.outputs.nodes {
			o.inputDeltas()
		}
	}

	n.calculateInputDeltas()
	n.completed = true
}

// assumes len(targets) == net.OutputSize()
func (net *Network) getDeltas(targets []float64) error {
	// initial conditions
	if net.stat < evaluated {
		return errors.Errorf("Internal error: Network has not been evaluated")
	}

	// reset deltas
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
	return nil
}

// only accurate while there is some calculation happening
func (net *Network) isGettingDeltas() bool {
	return net.stat >= evaluated
}

// assumes len(targets[n]) == net.OutputSize()
func (net *Network) adjustRecurrent(targets [][]float64, saveChanges bool) error {
	// Note: there was previously a line here, with unknown purpose:
	// net.stat = evaluated
	// with the comment:
	// // unsafe setting, but should practically be fine
	//
	// Additionally, the previous version bypassed getDeltas(), but this does not.

	if net.stat < finalized { // may run into issues if net.stat = finalized
		return errors.Errorf("Network is not complete")
	}

	for i := len(targets) - 1; i >= 0; i-- {
		if err := net.evaluate(true); err != nil {
			// this error should never occur
			return errors.Wrapf(err, "Failed to evaluate network with targets[%d]\n", i)
		}

		if err := net.getDeltas(targets[i]); err != nil {
			return errors.Wrapf(err, "Failed to get deltas of network with targets[%d]\n", i)
		}

		if err := net.adjust(true); err != nil {
			return errors.Wrapf(err, "Failed to adjust network with targets[%d]\n", i)
		}
	}

	if !saveChanges {
		net.AddWeights()
	}

	return nil
}

// does not use completion. There is no possibility of overlap
func (net *Network) adjust(saveChanges bool) error {
	if net.stat < deltas {
		return errors.Errorf("Internal error: Network has not had deltas calculated")
	}

	for _, n := range net.nodesByID {
		n.adjust(saveChanges)
	}

	if saveChanges {
		net.hasSavedChanges = true
	}

	return nil
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

	n.opt.Run(n, n.adj, w)
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

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
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
