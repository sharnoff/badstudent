package badstudent

import (
	"github.com/pkg/errors"
	// "math/rand"
)

type Network struct {
	inLayers  []*Layer
	outLayers []*Layer

	inputs  []float64
	outputs []float64

	layersByName map[string]*Layer
	layersByID   []*Layer
}

// dims, optimizer, inputs can be nil
func (net *Network) Add(name string, typ Operator, size int, dims []int, inputs ...*Layer) (*Layer, error) {

	if net.layersByName == nil {
		net.layersByName = make(map[string]*Layer)
	}

	if size < 1 {
		return nil, errors.Errorf("Can't add layer to network, layer must have >= 1 value (%d)", size)
	} else if net.layersByName[name] != nil {
		return nil, errors.Errorf("Can't add layer to network, name \"%s\" is already taken", name)
	} else if name == "" {
		return nil, errors.Errorf("Can't add layer to network, name cannot be \"\"")
	}

	l := new(Layer)
	l.Name = name
	l.status = initialized
	l.hostNetwork = net
	l.typ = typ
	l.id = len(net.layersByID)
	{
		l.dims = make([]int, len(dims))
		copy(l.dims, dims)
	}

	if len(inputs) == 0 {
		net.inLayers = append(net.inLayers, l)
	} else {
		l.inputs = inputs
		l.numInputs = make([]int, len(inputs))

		totalInputs := 0
		for i, in := range inputs {
			if in == nil {
				return nil, errors.Errorf("Fatal error: Can't add layer to network, input %d is nil (is there an extra argument to Add()?)", i)
			}

			totalInputs += in.Size()
			l.numInputs[i] = totalInputs

			if in.hostNetwork != net {
				return nil, errors.Errorf("Fatal error: Can't add layer to network, input %v (#%d) does not belong to the same *Network", in, i)
			}

			in.outputs = append(in.outputs, l)
		}
	}

	l.values = make([]float64, size)
	l.deltas = make([]float64, size)

	if err := typ.Init(l); err != nil {
		return nil, errors.Wrapf(err, "Fatal error: Couldn't add layer %v to network, initializing Operator failed\n", l)
	}

	net.layersByName[name] = l
	net.layersByID = append(net.layersByID, l)

	return l, nil
}

// checks that:
//  - no outputs are also inputs
//  - all given outputs belong to the network
//  - no layers don't affect the network outputs
func (net *Network) SetOutputs(outputs ...*Layer) error {
	if net.inLayers == nil {
		return errors.Errorf("Can't set outputs of network, network has no layers (net.inLayers == nil)")
	}

	numOutValues := 0
	for i, out := range outputs {
		if out.hostNetwork != net {
			return errors.Errorf("Can't set outputs of network, output layer %v (#%d) does not belong to this network", out, i)
		} else if len(out.inputs) == 0 {
			return errors.Errorf("Can't set outputs of network, layer %v (output %d) is both an input and an output", out, i)
		}

		out.isOutput = true
		out.placeInOutputs = numOutValues

		numOutValues += out.Size()
	}
	net.outLayers = outputs

	// remove the unused space at the end of slices
	{
		inputs := net.inLayers
		net.inLayers = make([]*Layer, len(inputs))
		copy(net.inLayers, inputs)
	}

	// allocate a single slice for the inputs and outputs, to make copying to and from them easier
	{
		net.outputs = make([]float64, numOutValues)
		place := 0
		for _, out := range net.outLayers {
			out.values = net.outputs[place : place+out.Size()]
			place += out.Size()
		}

		numInputs := 0
		for _, in := range net.inLayers {
			numInputs += in.Size()
		}

		net.inputs = make([]float64, numInputs)
		place = 0
		for _, in := range net.inLayers {
			in.values = net.inputs[place : place+in.Size()]
			place += in.Size()
		}
	}

	for i, in := range net.inLayers {
		if err := in.checkOutputs(); err != nil {
			return errors.Wrapf(err, "Can't set outputs of network, checking outputs of %v (input %d) failed\n", in, i)
		}
	}

	return nil
}
