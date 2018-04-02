package badstudent

import (
	"github.com/pkg/errors"
	"sync"
	"math"
	// "fmt"
)

type Layer struct {
	// The name that will be used to print this layer.
	// Not used for unique identification of any kind,
	// and may be nil
	Name string

	// the values of the layer -- essentially its outputs
	//
	// technically relates to the values after activation functions
	values  []float64

	// the derivative of each value w.r.t. the total cost
	// of the particular training example
	//
	// technically relates to the values before activation functions
	deltas  []float64 // δ

	// soon to be removed
	weights [][]float64

	input  *Layer
	output *Layer

	// whether or not the layer's values are part of the set of
	// output values to the network
	isOutput bool

	// keeps track of what has been calculated or completed for the layer
	status    status_

	// a lock for 'status'
	statusMux sync.Mutex
}

// soon to be removed
const bias_value float64 = 1

type status_ int8

const (
	// initialized  status_ = iota // 0
	checkOuts    status_ = iota // 1
	changed      status_ = iota // 2
	evaluated    status_ = iota // 3
	deltas       status_ = iota // 4
	adjusted     status_ = iota // 5
	// weightsAdded status_ = iota // 6
)

// returns the the Name of the Layer, surrounded by double quotes
func (l *Layer) String() string {
	return "\"" + l.Name + "\""
}

// checks that the Layer (and none of its outputs) don't affect
// the outputs of the network
//
// returns error if l.isOutput == false but len(l.outputs) == 0
func (l *Layer) checkOutputs() error {
	l.statusMux.Lock()
	defer l.statusMux.Unlock()
	if l.status >= checkOuts {
		return nil
	}

	if l.output == nil {
		if !l.isOutput {
			return errors.Errorf("Checked outputs; layer %v has no effect on network outputs\n", l)
		}
	} else if err := l.output.checkOutputs(); err != nil {
		return errors.Wrapf(err, "Checking outputs of layer %v from %v failed\n", l.output, l)
	}

	l.status = checkOuts
	return nil
}

// acts as a 'reset button' for the layer,
// signifying that it now will need to perform its operations again
//
// calls recursively on outputs
func (l *Layer) inputsChanged() {
	l.statusMux.Lock()
	if l.status < evaluated {
		l.statusMux.Unlock()
		return
	}

	l.status = changed
	l.statusMux.Unlock()

	if l.output != nil {
		l.output.inputsChanged()
	}
}

// updates the values of the layer so that they are accurate, given the inputs
//
// calls recursively on inputs before running
func (l *Layer) evaluate() {
	l.statusMux.Lock()
	defer l.statusMux.Unlock()
	if l.status >= evaluated {
		return
	} else if l.input == nil {
		l.status = evaluated
		return
	}

	if l.input != nil {
		l.input.evaluate()
	}

	for v := range l.values {
		var sum float64 // it may be better to just use l.values[v]
		for in := range l.weights[v] {
			if in != len(l.weights[v]) - 1 {
				sum += l.input.values[in] * l.weights[v][in]
			} else {
				sum += bias_value * l.weights[v][in]
			}
		}

		l.values[v] = 0.5 + 0.5*math.Tanh(0.5*sum)
	}

	l.status = evaluated
}

// calculates the deltas for each value of the layer
//
// calls inputDeltas() on outputs in order to run (which in turn calls getDeltas())
func (l *Layer) getDeltas(targets []float64) error {
	l.statusMux.Lock()
	defer l.statusMux.Unlock()
	if l.status < evaluated {
		return errors.Errorf("Can't get deltas of layer %v, has not been evaluated", l)
	} else if l.status >= deltas {
		return nil
	}

	if l.output != nil {
		deltaLocks := make([]sync.Mutex, len(l.deltas))

		add := func(index int, addition float64) {
			deltaLocks[index].Lock()
			l.deltas[index] += addition
			deltaLocks[index].Unlock()
		}

		l.deltas = make([]float64, len(l.deltas))

		if err := l.output.inputDeltas(l, add, targets); err != nil {
			return errors.Wrapf(err, "Can't get deltas of layer %v, input deltas of output failed\n", l)
		}
	} else {
		if len(targets) != len(l.values) {
			return errors.Errorf("Can't get deltas, len(targets) != len(l.values)")
		}

		for v := range l.values {
			l.deltas[v] = (l.values[v] - targets[v]) * l.values[v] * (1 - l.values[v])
		}
	}

	l.status = deltas
	return nil
}

// provides the deltas of each value to getDeltas()
//
// calls getDeltas() of self before running
func (l *Layer) inputDeltas(input *Layer, add func(int, float64), targets []float64) error {
	l.statusMux.Lock()
	if l.status < evaluated {
		l.statusMux.Unlock()
		return errors.Errorf("Can't provide input deltas of layer %v (to %v), has not been evaluated", l, input)
	}

	if l.status < deltas {
		// unlock status so that getDeltas() can lock it
		l.statusMux.Unlock()

		if err := l.getDeltas(targets); err != nil {
			return errors.Wrapf(err, "Can't provide input deltas of layer %v (to %v), getting own deltas failed\n", l, input)
		}

		l.statusMux.Lock()
	}

	// calculate the deltas, will be replaced
	// also handles tanh
	for in := range l.input.values {
		var sum float64
		for v := range l.values {
			sum += l.deltas[v] * l.weights[v][in]
		}

		// multiply to handle tanh
		sum *= l.input.values[in] * (1 - l.input.values[in])
		add(in, sum)
	}

	l.statusMux.Unlock()
	return nil
}

// recurses to inputs after running
// α
func (l *Layer) adjust(learningRate float64) error {
	l.statusMux.Lock()
	if l.status < deltas {
		l.statusMux.Unlock()
		return errors.Errorf("Can't adjust layer %v, has not calculated deltas", l)
	} else if l.status >= adjusted {
		l.statusMux.Unlock()
		return nil
	} else if l.input == nil {
		l.status = adjusted
		l.statusMux.Unlock()
		return nil
	}

	for v := range l.deltas {
		for w := range l.weights[v] {
			// the gradient is: l.input.values[w] * l.deltas[v]
			if w != len(l.weights[v]) - 1 {
				l.weights[v][w] += -1 * learningRate * l.input.values[w] * l.deltas[v]
			} else {
				l.weights[v][w] += -1 * learningRate * bias_value * l.deltas[v]
			}
		}
	}

	l.status = adjusted
	l.statusMux.Unlock()

	if l.input != nil {
		if err := l.input.adjust(learningRate); err != nil {
			return errors.Wrapf(err, "Failed to recurse after adjusting\n")
		}
	}

	l.inputsChanged()

	return nil
}
