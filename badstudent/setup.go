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

	if len(net.inLayers) == 0 {
		net.inLayers = make([]*Layer, 1)
	}
	if len(net.outLayers) == 0 {
		net.outLayers = make([]*Layer, 1)
	}

	l := new(Layer)
	l.Name = name
	l.status = changed
	l.isOutput = true

	if net.outLayers[0] != nil {
		l.input = net.outLayers[0]
		net.outLayers[0].output = l
		
		l.input.isOutput = false
	} else {
		net.inLayers[0] = l
	}
	net.outLayers[0] = l

	l.values = make([]float64, size)
	l.deltas = make([]float64, size)
	l.initWeights()

	return nil
}

func (net *Network) SetOutputs() error {
	if net.inLayers[0] == nil {
		return errors.Errorf("Can't set outputs of network, network has no layers (net.inLayers[0] == nil)")
	} else if net.inLayers[0] == net.outLayers[0] {
		return errors.Errorf("Can't set outputs of network, input and output are identical (%v)", net.inLayers[0])
	}

	if err := net.inLayers[0].checkOutputs(); err != nil {
		return errors.Wrapf(err, "Can't set outputs of network, checking outputs of %v failed\n", net.inLayers[0])
	}

	return nil
}
