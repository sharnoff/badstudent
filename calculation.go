package badstudent

// internal management of the steps of evaluation and correction

import (
	"github.com/pkg/errors"
)

type status_ int8

const (
	initialized  status_ = iota // 0
	checkOuts    status_ = iota // 1
	changed      status_ = iota // 2
	evaluated    status_ = iota // 3
	deltas       status_ = iota // 4
	adjusted     status_ = iota // 5
	weightsAdded status_ = iota // 6
)

// soon to be removed
const bias_value float64 = 1

// checks that the Node (and none of its outputs) don't affect
// the outputs of the network
//
// returns error if n.isOutput == false but len(n.outputs) == 0
//
// recurses towards outputs
func (n *Node) checkOutputs() error {
	n.statusMux.Lock()
	defer n.statusMux.Unlock()
	if n.status >= checkOuts {
		return nil
	}

	if len(n.outputs) == 0 && !n.isOutput {
		return errors.Errorf("Checked outputs; node %v has no effect on network outputs (has no outputs and is not a network output)\n", n)
	}

	// remove the unused space at the end of slices
	{
		outputs := n.outputs
		n.outputs = make([]*Node, len(outputs))
		copy(n.outputs, outputs)
	}

	for i, out := range n.outputs {
		if err := out.checkOutputs(); err != nil {
			return errors.Wrapf(err, "Checking outputs of output %d to node %v (%v) failed\n", i, n, out)
		}
	}

	n.status = checkOuts
	return nil
}

// checks the outputs of all nodes in the network
func (net *Network) checkOutputs() error {
	for i, in := range net.inputs.nodes {
		if err := in.checkOutputs(); err != nil {
			return errors.Wrapf(err, "Failed to check outputs of network input %v (#%d)\n", in, i)
		}
	}

	return nil
}

// acts as a 'reset button' for the node,
// signifying that it now will need to perform its operations again
//
// calls recursively on outputs
func (n *Node) inputsChanged() {
	n.statusMux.Lock()
	if n.status < evaluated {
		n.statusMux.Unlock()
		return
	}

	n.status = changed
	n.statusMux.Unlock()

	for _, out := range n.outputs {
		out.inputsChanged()
	}
}

// sets the inputs of the network to the provided values
// returns an error if the length of the provided values doesn't
// match the size of the network inputs
func (net *Network) SetInputs(inputs []float64) error {
	return net.inputs.setValues(inputs)
}

// updates the values of the node so that they are accurate, given the inputs
//
// calls recursively on inputs before running
func (n *Node) evaluate() error {
	n.statusMux.Lock()
	defer n.statusMux.Unlock()
	if n.status >= evaluated && n.status != weightsAdded {
		return nil
	} else if len(n.inputs) == 0 {
		n.status = evaluated
		return nil
	}

	for i, in := range n.inputs {
		if err := in.evaluate(); err != nil {
			return errors.Wrapf(err, "Can't evaluate node %v, evaluating input %v (#%d) failed\n", n, in, i)
		}
	}

	if err := n.typ.Evaluate(n, n.values); err != nil {
		return errors.Wrapf(err, "Couldn't evaluate node %v, Operation evaluation failed\n", n)
	}

	n.status = evaluated
	return nil
}

// Returns a copy of the output values of the Network, given the inputs
//
// Returns an error if it can't the given inputs to be the network's
func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if err := net.SetInputs(inputs); err != nil {
		return nil, errors.Wrapf(err, "Couldn't get outputs; setting inputs failed.\n")
	}

	for i, out := range net.outputs.nodes {
		if err := out.evaluate(); err != nil {
			return nil, errors.Wrapf(err, "Can't get outputs, network output node %v (#%d) failed to evaluate\n", out, i)
		}
	}

	return net.outputs.getValues(true), nil
}

// Calculates the deltas for each value of the node
//
// Calls inputDeltas() on outputs in order to run (which in turn calls getDeltas())
// deltasMatter is: do the deltas of this node actually need to be calculated, or should this
// just pass the recursion to its outputs
func (n *Node) getDeltas(rangeCostDeriv func(int, int, func(int, float64)) error, deltasMatter bool) error {

	deltasMatter = deltasMatter || n.typ.CanBeAdjusted(n)

	n.statusMux.Lock()
	defer n.statusMux.Unlock()
	if n.status < evaluated {
		return errors.Errorf("Can't get deltas of node %v, has not been evaluated", n)
	} else if n.status >= deltas && !(deltasMatter && !n.deltasActuallyCalculated) { // REWORK
		return nil
	}

	if !deltasMatter {
		for i, out := range n.outputs {
			if err := out.getDeltas(rangeCostDeriv, deltasMatter); err != nil { // deltasMatter = false
				return errors.Wrapf(err, "Can't pass on getting deltas from node %v, getting deltas of node %v (output %d) failed\n", n, out, i)
			}
		}
	} else {
		add := func(index int, addition float64) {
			n.deltas[index] += addition
		}

		// reset all of the values
		n.deltas = make([]float64, len(n.values))

		if n.isOutput {
			if err := rangeCostDeriv(0, len(n.values), add); err != nil {
				return errors.Wrapf(err, "Can't get deltas of output node %v, rangeCostDeriv() failed\n", n)
			}
		}

		for i, out := range n.outputs {
			if err := out.inputDeltas(n, add, rangeCostDeriv); err != nil {
				return errors.Wrapf(err, "Can't get deltas of node %v, input deltas from node %v (output %d) failed\n", n, out, i)
			}
		}

		n.deltasActuallyCalculated = true
	}

	n.status = deltas
	return nil
}

// provides the deltas of each value to getDeltas()
//
// calls getDeltas() of self before running
func (n *Node) inputDeltas(input *Node, add func(int, float64), rangeCostDeriv func(int, int, func(int, float64)) error) error {
	n.statusMux.Lock()
	if n.status < evaluated {
		n.statusMux.Unlock()
		return errors.Errorf("Can't provide input deltas of node %v (to %v), has not been evaluated", n, input)
	}

	if n.status < deltas {
		// unlock status so that getDeltas() can lock it
		n.statusMux.Unlock()

		if err := n.getDeltas(rangeCostDeriv, true); err != nil { // deltasMatter = true
			return errors.Wrapf(err, "Can't provide input deltas of node %v (to %v), getting own deltas failed\n", n, input)
		}

		n.statusMux.Lock()
	}

	// find the index in 'n.inputs' that 'input' is. If not there, return error
	inputIndex := -1
	for i := range n.inputs {
		if n.inputs[i] == input {
			inputIndex = i
			break
		}
	}

	if inputIndex == -1 {
		return errors.Errorf("Can't provide input deltas of node %v to %v, %v is not an input of %v", n, input, input, n)
	}

	start := n.PreviousInputs(inputIndex)
	end := start + n.InputSize(inputIndex)

	if err := n.typ.InputDeltas(n, add, start, end); err != nil {
		return errors.Wrapf(err, "Couldn't provide input deltas of node %v to %v (#%d), Operator failed to get input deltas\n", n, input, inputIndex)
	}

	n.statusMux.Unlock()
	return nil
}

// recurses to inputs after running
// α
func (n *Node) adjust(learningRate float64, saveChanges bool) error {
	n.statusMux.Lock()
	if n.status < deltas {
		n.statusMux.Unlock()
		return errors.Errorf("Can't adjust node %v, has not calculated deltas", n)
	} else if n.status >= adjusted {
		n.statusMux.Unlock()
		return nil
	} else if n.inputs == nil {
		n.status = adjusted
		n.statusMux.Unlock()
		return nil
	}

	if err := n.typ.Adjust(n, learningRate, saveChanges); err != nil {
		return errors.Wrapf(err, "Couldn't adjust node %v, Operator adjusting failed\n", n)
	}

	n.status = adjusted
	n.statusMux.Unlock()

	for i, in := range n.inputs {
		if err := in.adjust(learningRate, saveChanges); err != nil {
			return errors.Wrapf(err, "Failed to recurse after adjusting weights to node %v (input %d) from node %v\n", in, i, n)
		}
	}

	return nil
}

// recurses to inputs after running
func (n *Node) addWeights() error {
	n.statusMux.Lock()
	if n.status >= weightsAdded {
		n.statusMux.Unlock()
		return nil
	}

	if err := n.typ.AddWeights(n); err != nil {
		return errors.Wrapf(err, "Couldn't add weights for node %v, Operator failed to add weights\n", n)
	}

	n.status = weightsAdded
	n.statusMux.Unlock()

	for i, in := range n.inputs {
		if err := in.addWeights(); err != nil {
			return errors.Wrapf(err, "Failed to recurse to %v (input %d) after adding weights of node %v\n", in, i, n)
		}
	}

	return nil
}

// Updates the weights in the newtork with any previously delayed changes
func (net *Network) AddWeights() error {

	for i, out := range net.outputs.nodes {
		if err := out.addWeights(); err != nil {
			return errors.Wrapf(err, "Couldn't add weights of network, output node %v (#%d) failed to add weights\n", out, i)
		}
	}

	return nil
}
