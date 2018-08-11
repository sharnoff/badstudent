package badstudent

import (
	"github.com/pkg/errors"
	"strings"
)

func (net *Network) init() {
	if net.nodesByName != nil {
		return
	}

	net.nodesByName = make(map[string]*Node)
	net.inputs = new(nodeGroup)
}

// Adds a new node to the Network, with given name, size, inputs, and Operator
// If no inputs are given, the node will be one of the input nodes, and its size added to the
// number of inputs
//
// The name of each node must be unique, cannot be "", and cannot contain a double-quote (")
//
// if Add returns an error, the host Network will not have been changed
func (net *Network) Add(name string, typ Operator, size, delay int, inputs ...*Node) (*Node, error) {
	n, err := net.Placeholder(name, size)
	if err != nil {
		return n, err
	}

	if err := n.Replace(typ, delay, inputs...); err != nil {
		net.nodesByName[name] = nil

		copy(net.nodesByID[n.id:], net.nodesByID[n.id + 1:])
		net.nodesByID = net.nodesByID[:len(net.nodesByID)-1]

		return n, err
	}

	return n, nil
}

// Returns a placeholder Node that can be used as input, later to have its own inputs set
// This Node must have been replaced before the outputs of the network can be set
func (net *Network) Placeholder(name string, size int) (*Node, error) {
	net.init()

	if size < 1 {
		return nil, errors.Errorf("Node must have size >= 1 (%d)", size)
	} else if net.nodesByName[name] != nil {
		return nil, errors.Errorf("Name %q is already taken", name)
	} else if name == "" {
		return nil, errors.Errorf(`Name cannot be ""`)
	} else if strings.Contains(name, `"`) {
		return nil, errors.Errorf(`Name contains illegal character:"`)
	}

	n := new(Node)
	n.name = name
	n.host = net
	n.id = len(net.nodesByID)
	n.placeInOutputs = -1

	n.outputs = new(nodeGroup)

	n.values = make([]float64, size)
	n.deltas = make([]float64, size)

	net.nodesByName[name] = n
	net.nodesByID = append(net.nodesByID, n)

	return n, nil
}

// Sets the inputs and Operator of a placeholder Node
//
// Network must still be in construction
func (n *Node) Replace(typ Operator, delay int, inputs ...*Node) error {
	if n.host.stat > initialized {
		return errors.Errorf("Network has finished construction")
	} else if !n.IsPlaceholder() {
		return errors.Errorf("Node is not a placeholder")
	} else if typ == nil {
		return errors.Errorf("Operator is nil")
	} else if delay < 0 {
		return errors.Errorf("Node must have delay of 0 or above")
	} else if len(inputs) == 0 && delay > 0 {
		return errors.Errorf("Input nodes cannot have delay (delay = %d)", delay)
	}

	for _, in := range inputs {
		if in.id > n.id {
			n.host.mayHaveLoop = true
		}
	}

	for i, in := range inputs {
		if in == nil {
			return errors.Errorf("Input %d to node %v is nil", i)
		} else if in.host != n.host {
			return errors.Errorf("Input %d (%v) to %v does not belong to the same Network", i, in, n)
		}
	}

	n.inputs = new(nodeGroup)
	n.inputs.add(inputs...)

	if err := typ.Init(n); err != nil {
		return errors.Wrapf(err, "Initializing Operator failed\n", n)
	}

	n.typ = typ

	n.delay = make(chan []float64, delay)
	n.delayDeltas = make(chan []float64, delay)
	for i := 0; i < delay; i++ {
		n.delay <- make([]float64, len(n.values))
		n.delayDeltas <- make([]float64, len(n.values))
	}

	if delay != 0 {
		n.host.hasDelay = true
	}

	if len(inputs) == 0 {
		n.host.inputs.add(n)
	}

	for _, in := range inputs {
		in.outputs.add(n)
	}

	n.completed = true
	return nil
}

// Finalizes the structure of the Network.
//
// No outputs can be inputs
// All Nodes must affect the outputs
//
// If an error is returned, the Network has remained unchanged
func (net *Network) SetOutputs(outputs ...*Node) error {
	if len(net.nodesByID) == 0 {
		return errors.Errorf("Can't set outputs of network, network has no nodes")
	} else if len(outputs) == 0 {
		return errors.Errorf("Can't set outputs of network, none given")
	}

	// Removed the completion marks by Add() and Replace()
	net.resetCompletion()

	// if setting outputs needs to be aborted, the nodes should not be affected by '.isOutput'
	// being true. This sets them back to false for those nodes that may have already been
	// switched
	var allGood = false
	defer func() {
		if !allGood {
			for _, out := range outputs {
				if out != nil {
					out.placeInOutputs = -1
				}
			}
		}
	}()

	for i, out := range outputs {
		if out == nil {
			return errors.Errorf("Can't set outputs of network, output node #%d is nil, i")
		} else if out.host != net {
			return errors.Errorf("Can't set outputs of network, output node #%d (%v) does not belong to this network", i, out)
		} else if num(out.inputs) == 0 {
			return errors.Errorf("Can't set outputs of network, output node #%d (%v) is both an input and an output", i, out)
		}

		// check that there are no duplicates
		for o := i + 1; o < len(outputs); o++ {
			if out == outputs[o] {
				return errors.Errorf("Can't set outputs of network, output #%d (%v) is also #%d", i, out, o)
			}
		}

		out.placeInOutputs = 0 // mark as having a place in outputs
	}

	net.outputs = new(nodeGroup)
	net.outputs.add(outputs...)

	if err := net.checkOutputs(); err != nil {
		return err
	}

	allGood = true

	numOutValues := 0
	for _, out := range outputs {
		out.placeInOutputs = numOutValues

		numOutValues += out.Size()
	}

	// Slightly reduce memory usage
	for _, n := range net.nodesByID {
		n.outputs.trim()
	}

	// remove the unused space at the end of slices
	net.inputs.trim()

	// allocate single slices for inputs and outputs
	net.inputs.makeContinuous()
	net.outputs.makeContinuous()

	net.stat = finalized

	return nil
}
