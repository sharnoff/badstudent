package badstudent

import (
	"github.com/pkg/errors"
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

// clears delay-related functions for all nodes in the network
// Essentially flushes a recurrent-type network
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
// * each node's need for having its own deltas calculated
func (net *Network) finalize() error {
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
			} else if root.Delay() != 0 {
				// This node outputs to everything with delay, so it cannot
				// output to itself with none
				continue
			}

			check = func(n *Node) error {
				if n.completed {
					return nil
				} else if n.Delay() != 0 {
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

	// determine each node's need for having deltas calculated
	{
		var matter func(*Node, bool)
		matter = func(n *Node, deltasMatter bool) {
			// condition for running
			if !n.completed || (!n.deltasMatter && deltasMatter) {

				if n.IsInput() {
					deltasMatter = false
				} else {
					deltasMatter = deltasMatter || n.typ.CanBeAdjusted(n)
				}
				n.deltasMatter = deltasMatter
				n.completed = true

				for _, out := range n.outputs.nodes {
					matter(out, n.deltasMatter)
				}
			}

			return
		}

		for _, in := range net.inputs.nodes {
			matter(in, false)
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

// A full diagram of the operational flow of evaluate() and getDeltas() is available upon request
func (n *Node) evaluate() error {
	if n.completed {
		return nil
	} else if n.IsInput() {
		if n.host.hasDelay {
			if n.host.isGettingDeltas() {
				n.setValues(n.storedValues[len(n.storedValues)-1])
				n.storedValues = n.storedValues[:len(n.storedValues)-1]
			} else {
				vs := n.values
				if n.group != nil { // if n's values will be overwritten
					vs = make([]float64, len(n.values))
					copy(vs, n.values)
				}

				n.storedValues = append(n.storedValues, vs)
			}
		}

		n.completed = true
		return nil
	}

	if n.HasDelay() {
		if !n.host.isGettingDeltas() {
			n.setValues(<-n.delay)
		} else { // if getting deltas
			n.setValues(n.storedValues[len(n.storedValues)-1])
			n.storedValues = n.storedValues[:len(n.storedValues)-1]
		}
		n.completed = true
	}

	for _, in := range n.inputs.nodes {
		if err := in.evaluate(); err != nil {
			return errors.Wrapf(err, "Evaluating Node %v failed\n", in)
		}
	}

	values := n.values
	if n.HasDelay() {
		values = make([]float64, len(n.values))
	}

	if err := n.typ.Evaluate(n, values); err != nil {
		return errors.Wrapf(err, "Operator evaluation failed\n")
	}

	if n.HasDelay() && !n.host.isGettingDeltas() {
		n.delay <- values
		n.storedValues = append(n.storedValues, values)
	}

	n.completed = true
	return nil
}

// Changes the values of the Nodes so that they accurately reflect the inputs
//
// 'recurrent' forces ignoring avoiding repetition if net.stat >= evaluated, if true
func (net *Network) evaluate(recurrent bool) error {
	if net.stat < finalized {
		return errors.Errorf("Network is not complete")
	} else if !recurrent && net.stat >= evaluated {
		return nil
	}

	for _, out := range net.outputs.nodes {
		if err := out.evaluate(); err != nil {
			return errors.Wrapf(err, "Evaluating Network output Node %v failed\n", out)
		}
	}

	net.resetCompletion()
	net.stat = evaluated

	return nil
}

// Calculates the deltas for each value of the node
//
// A full diagram of the operational logic of evaluate() and getDeltas() is available upon request
func (n *Node) getDeltas(cfDeriv func(int, int, func(int, float64)) error) error {

	if n.completed && !n.HasDelay() {
		return nil
	} else if !n.deltasMatter {
		n.completed = true

		for _, out := range n.outputs.nodes {
			if err := out.getDeltas(cfDeriv); err != nil {
				return errors.Wrapf(err, "Getting deltas of Node %v failed\n", out)
			}
		}

		return nil
	}

	deltas := make([]float64, len(n.values))

	add := func(index int, addition float64) {
		deltas[index] += addition
	}

	if n.IsOutput() {
		if err := cfDeriv(n.placeInOutputs, n.placeInOutputs+len(n.values), add); err != nil {
			return errors.Wrapf(err, "Getting derivatives of cost function failed\n")
		}
	}

	for _, out := range n.outputs.nodes {
		if err := out.inputDeltas(n, add, cfDeriv); err != nil {
			return errors.Wrapf(err, "Getting input deltas of Node %v from Node %v failed\n", out, n)
		}
	}

	if n.HasDelay() {
		n.delayDeltas <- deltas
	} else {
		n.deltas = deltas
	}

	n.completed = true
	return nil
}

// provides the deltas of each value to getDeltas()
// calls getDeltas() of self before running
//
// Doesn't mark complete or not because this could be called multiple times for multiple inputs
func (n *Node) inputDeltas(input *Node, add func(int, float64), cfDeriv func(int, int, func(int, float64)) error) error {

	// if it's done, don't bother getting deltas
	// This is needed to avoid secondary loops
	if !n.completed {
		if n.HasDelay() {
			// this case should never happen, but it's here for security
			n.deltas = <-n.delayDeltas
			n.completed = true
		} else {
			if err := n.getDeltas(cfDeriv); err != nil {
				return errors.Errorf("Getting deltas of Node %v failed\n", n)
			}
		}
	}

	// find the index in the inputs of 'n' that 'input' is.
	inputIndex := -1
	for i, in := range n.inputs.nodes {
		if in == input {
			inputIndex = i
			break
		}
	}
	if inputIndex == -1 {
		return errors.Errorf("Can't provide input deltas of node %v to %v, %v is not an input of %v", n, input, input, n)
	}

	end := n.inputs.sumVals[inputIndex]
	start := end - input.Size()

	if err := n.typ.InputDeltas(n, add, start, end); err != nil {
		return errors.Wrapf(err, "Operator input delta calculation failed\n")
	}

	return nil
}

// only accurate while there is some calculation happening
func (net *Network) isGettingDeltas() bool {
	return net.stat >= evaluated
}

// Calculates the deltas of all of the Nodes in the Network whose deltas
// must be calculated
//
// targets is assumed to be the proper length
func (net *Network) getDeltas(targets []float64, cf CostFunction) error {
	if net.stat < evaluated {
		return errors.Errorf("Network must be evaluated before getting deltas")
	} else if net.stat >= deltas {
		return nil
	} else if net.hasDelay {
		return errors.Errorf("Cannot obtain deltas for Network with delay")
	}

	cfDeriv := func(start, end int, add func(int, float64)) error {
		return cf.Deriv(net.outputs.getValues(false), targets, start, end, add)
	}

	for i, in := range net.inputs.nodes {
		if err := in.getDeltas(cfDeriv); err != nil {
			return errors.Wrapf(err, "Failed to get deltas for network input %v (#%d)\n", in, i)
		}
	}

	net.stat = deltas
	net.resetCompletion()
	return nil
}

func (net *Network) adjustRecurrent(targets [][]float64, cf CostFunction, learningRates []float64, saveChanges bool) error {
	// unsafe setting, but should practically be fine
	net.stat = evaluated

	for i := len(targets) - 1; i >= 0; i-- {
		if err := net.evaluate(true); err != nil {
			return errors.Wrapf(err, "Failed to evaluate network to use targets %d\n", i)
		}

		cfDeriv := func(start, end int, add func(int, float64)) error {
			return cf.Deriv(net.outputs.getValues(false), targets[i], start, end, add)
		}

		// bypass getDeltas
		{
			for _, n := range net.nodesByID {
				if n.HasDelay() && n.deltasMatter {
					n.deltas = <-n.delayDeltas
					n.completed = true
				}
			}

			for i, in := range net.inputs.nodes {
				if err := in.getDeltas(cfDeriv); err != nil {
					return errors.Wrapf(err, "Failed to get deltas for network input %v (#%d)\n", in, i)
				}
			}

			for _, n := range net.nodesByID {
				if n.HasDelay() {
					if err := n.getDeltas(cfDeriv); err != nil {
						return errors.Wrapf(err, "Failed to get deltas for node with delay: %v\n", n)
					}
				}
			}

			net.stat = deltas
			net.resetCompletion()
		}

		if err := net.adjust(learningRates[i], true); err != nil {
			return errors.Wrapf(err, "Failed to adjust after getting deltas with targets[%d]\n", i)
		}
	}

	if !saveChanges {
		if err := net.AddWeights(); err != nil {
			return errors.Wrapf(err, "Failed to add weights after adjusting\n")
		}
	}

	net.ClearDelays()
	return nil
}

func (n *Node) adjust(learningRate float64, saveChanges bool) error {
	if n.IsInput() {
		return nil
	}

	if err := n.typ.Adjust(n, learningRate, saveChanges); err != nil {
		return errors.Wrapf(err, "Operator adjustment failed\n")
	}

	return nil
}

// Does not use completion to track progress
func (net *Network) adjust(learningRate float64, saveChanges bool) error {
	if net.stat < deltas {
		return errors.Errorf("Network must have deltas calculated before adjusting weights")
	}

	for _, n := range net.nodesByID {
		if err := n.adjust(learningRate, saveChanges); err != nil {
			return errors.Wrapf(err, "Failed to adjust Node %v\n", n)
		}
	}

	if saveChanges {
		net.hasSavedChanges = true
	}

	net.stat = adjusted
	return nil
}

func (n *Node) addWeights() error {
	if n.IsInput() {
		return nil
	}

	if err := n.typ.AddWeights(n); err != nil {
		return errors.Wrapf(err, "Operator failed to add weights\n", n)
	}

	return nil
}

// Updates the weights in the network with any previously saved changes.
// Only runs if there are changes that have not been applied
func (net *Network) AddWeights() error {
	if !net.hasSavedChanges {
		return nil
	}

	for _, n := range net.nodesByID {
		if err := n.addWeights(); err != nil {
			return errors.Wrapf(err, "Failed to add weights of Node %v\n", n)
		}
	}

	net.hasSavedChanges = false
	net.stat = finalized
	return nil
}
