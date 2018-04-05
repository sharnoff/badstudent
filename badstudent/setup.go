package badstudent

import (
	"github.com/pkg/errors"
	"math/rand"
)

type Network struct {
	inLayers  []*Layer
	outLayers []*Layer

	inputs  []float64
	outputs []float64
}

func (l *Layer) initWeights() {
	if l.inputs == nil {
		return
	}

	numInputs := 0
	for _, in := range l.inputs {
		numInputs += len(in.values)
	}

	l.weights = make([][]float64, len(l.values))
	for v := range l.weights {
		l.weights[v] = make([]float64, numInputs + 1) // +1 for bias
		for in := range l.weights[v] {
			l.weights[v][in] = (2 * rand.Float64() - 1) / float64(numInputs + 1)
		}
	}
}

func (net *Network) Add(name string, size int, inputs ...*Layer) (*Layer, error) {
	if size < 1 {
		return nil, errors.Errorf("Can't add layer to network, layer must have >= 1 values (%d)", size)
	}

	l := new(Layer)
	l.Name = name
	l.status = initialized
	l.hostNetwork = net

	if len(inputs) == 0 {
		net.inLayers = append(net.inLayers, l)
	} else {
		l.inputs = inputs
		l.numInputs = make([]int, len(inputs))
		
		totalInputs := 0
		for i, in := range inputs {
			totalInputs += in.Size()
			l.numInputs[i] = totalInputs

			if in.hostNetwork != net {
				return nil, errors.Errorf("Can't add layer to network, input %v (#%d) does not belong to the same *Network", in, i)
			}

			in.outputs = append(in.outputs, l)
		}
	}

	l.values = make([]float64, size)
	l.deltas = make([]float64, size)
	l.initWeights()

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
			out.values = net.outputs[ place : place + out.Size() ]
			place += out.Size()
		}

		numInputs := 0
		for _, in := range net.inLayers {
			numInputs += in.Size()
		}

		net.inputs = make([]float64, numInputs)
		place = 0
		for _, in := range net.inLayers {
			in.values = net.inputs[ place : place + in.Size() ]
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
