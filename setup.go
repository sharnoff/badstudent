package badstudent

import (
	"github.com/pkg/errors"
	"strings"
)

// Adds a new node to the Network, with given name, size, inputs, and Operator
// If no inputs are given, the node will be one of the input nodes, and its size added to the
// number of inputs
//
// The name of each node must be unique, cannot be "", and cannot contain a `"`
//
// if Add returns an error, the host Network will not be functionally different
func (net *Network) Add(name string, typ Operator, size int, inputs ...*Node) (*Node, error) {

	if net.nodesByName == nil {
		net.nodesByName = make(map[string]*Node)
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

	l := new(Node)
	l.Name = name
	l.status = initialized
	l.hostNetwork = net
	l.typ = typ
	l.id = len(net.nodesByID)

	{
		l.inputs = inputs
		l.numInputs = make([]int, len(inputs))
		totalInputs := 0
		for i, in := range inputs {
			if in == nil {
				return nil, errors.Errorf("Can't add node to network, input #%d is nil", i)
			} else if in.hostNetwork != net {
				return nil, errors.Errorf("Can't add node to network, input #%d (%v) does not belong to the same Network", i, in)
			}

			totalInputs += in.Size()
			l.numInputs[i] = totalInputs
		}
	}

	l.values = make([]float64, size)
	l.deltas = make([]float64, size)

	if err := typ.Init(l); err != nil {
		return nil, errors.Wrapf(err, "Couldn't add node %v to network, initializing Operator failed\n", l)
	}

	if len(inputs) == 0 {
		net.inNodes = append(net.inNodes, l)
	} else {
		for _, in := range inputs {
			in.outputs = append(in.outputs, l)
		}
	}

	net.nodesByName[name] = l
	net.nodesByID = append(net.nodesByID, l)

	return l, nil
}

// Finalizes the Network, checking that:
//  - no output Nodes are also inputs
//  - all given outputs belong to the given network
//  - all nodes affect the outputs
//
// if SetOutputs returns an error, the network has remained unchanged
func (net *Network) SetOutputs(outputs ...*Node) error {
	if net.inNodes == nil {
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
		} else if out.hostNetwork != net {
			return errors.Errorf("Can't set outputs of network, output node #%d (%v) does not belong to this network", i, out)
		} else if len(out.inputs) == 0 {
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

	if err := net.checkOutputs(); err != nil {
		return err
	}

	allGood = true

	numOutValues := 0
	for _, out := range outputs {
		out.placeInOutputs = numOutValues

		numOutValues += out.Size()
	}
	net.outNodes = outputs

	// remove the unused space at the end of slices
	{
		inputs := net.inNodes
		net.inNodes = make([]*Node, len(inputs))
		copy(net.inNodes, inputs)
	}

	// allocate a single slice for the inputs and outputs, to make copying to and from them easier
	{
		net.outputs = make([]float64, numOutValues)
		place := 0
		for _, out := range net.outNodes {
			out.values = net.outputs[place : place+out.Size()]
			place += out.Size()
		}

		numInputs := 0
		for _, in := range net.inNodes {
			numInputs += in.Size()
		}

		net.inputs = make([]float64, numInputs)
		place = 0
		for _, in := range net.inNodes {
			in.values = net.inputs[place : place+in.Size()]
			place += in.Size()
		}
	}

	return nil
}
