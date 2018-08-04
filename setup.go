package badstudent

import (
	"github.com/pkg/errors"
	"strings"
)

// Adds a new node to the Network, with given name, size, inputs, and Operator
// If no inputs are given, the node will be one of the input nodes, and its size added to the
// number of inputs
//
// The name of each node must be unique, cannot be empty, and cannot contain a `"`
//
// if Add returns an error, the host Network will not be functionally different
func (net *Network) Add(name string, typ Operator, size int, inputs ...*Node) (*Node, error) {

	// if the network has not been initialized
	if net.nodesByName == nil {
		net.nodesByName = make(map[string]*Node)
		net.inputs = new(nodeGroup)
	}

	if size < 1 {
		return nil, errors.Errorf("Can't add node to network, node must have >= 1 value (%d)", size)
	} else if net.nodesByName[name] != nil {
		return nil, errors.Errorf("Can't add node to network, name \"%s\" is already taken", name)
	} else if name == "" {
		return nil, errors.Errorf("Can't add node to network, name cannot be \"\"")
	} else if strings.Contains(name, `"`) {
		return nil, errors.Errorf("Can't add node to network, name cannot contain \"")
	}

	n := new(Node)
	n.name = name
	n.host = net
	n.typ = typ
	n.id = len(net.nodesByID)

	n.inputs = new(nodeGroup)
	n.outputs = new(nodeGroup)

	n.inputs.add(inputs...)

	for i, in := range inputs {
		if in == nil {
			return nil, errors.Errorf("Can't add node to network, input #%d is nil", i)
		} else if in.host != net {
			return nil, errors.Errorf("Can't add node to network, input #%d (%v) does not belong to the same Network", i, in)
		}
	}

	n.values = make([]float64, size)
	n.deltas = make([]float64, size)

	if err := typ.Init(n); err != nil {
		return nil, errors.Wrapf(err, "Couldn't add node %v to network, initializing Operator failed\n", n)
	}

	if len(inputs) == 0 {
		net.inputs.add(n)
	} else {
		for _, in := range inputs {
			in.outputs.add(n)
		}
	}

	net.nodesByName[name] = n
	net.nodesByID = append(net.nodesByID, n)

	return n, nil
}

// Finalizes the Network, checking that:
//  - no output Nodes are also inputs
//  - all given outputs belong to the given network
//  - all nodes affect the outputs
//
// if SetOutputs returns an error, the network has remained unchanged
func (net *Network) SetOutputs(outputs ...*Node) error {
	if len(net.nodesByID) == 0 {
		return errors.Errorf("Can't set outputs of network, network has no nodes")
	} else if len(outputs) == 0 {
		return errors.Errorf("Can't set outputs of network, none given")
	}

	// if setting outputs needs to be aborted, the nodes should not be affected by '.isOutput'
	// being true. This sets them back to false for those nodes that may have already been
	// switched
	var allGood = false
	defer func() {
		if !allGood {
			for _, out := range outputs {
				if out != nil {
					out.isOutput = false
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

		out.isOutput = true
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
