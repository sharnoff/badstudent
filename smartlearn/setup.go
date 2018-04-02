package smartlearn

import (
	"github.com/pkg/errors"
	"math/rand"
)

type Network struct {
	input *Layer
	output *Layer
}

func (l *Layer) initWeights() {
	numInputs := 0
	if l.input != nil {
		numInputs = len(l.input.values)
	}

	l.weights = make([][]float64, len(l.values))
	for v := range l.weights {
		l.weights[v] = make([]float64, numInputs + 1) // +1 for bias
		for in := range l.weights[v] {
			l.weights[v][in] = (2 * rand.Float64() - 1) / float64(numInputs + 1)
		}
	}
}

func (net *Network) Add(name string, size int) error {
	if size < 1 {
		return errors.Errorf("Can't add layer to network, layer must have >= 1 neurons (%d)", size)
	}

	l := new(Layer)
	l.name = name
	l.status = changed
	l.isOutput = true

	if net.output != nil {
		l.input = net.output
		net.output.output = l
		
		l.input.isOutput = false
	} else {
		net.input = l
	}
	net.output = l

	l.values = make([]float64, size)
	l.deltas = make([]float64, size)
	l.initWeights()

	return nil
}

func (net *Network) SetOutputs() error {
	if net.input == nil {
		return errors.Errorf("Can't set outputs of network, network has no layers (net.input == nil)")
	} else if net.input == net.output {
		return errors.Errorf("Can't set outputs of network, input and output are identical (%v)", net.input)
	}

	if err := net.input.checkOutputs(); err != nil {
		return errors.Wrapf(err, "Can't set outputs of network, checking outputs of %v failed\n", net.input)
	}

	return nil
}
