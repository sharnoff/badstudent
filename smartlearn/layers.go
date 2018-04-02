package smartlearn

import (
	"github.com/pkg/errors"
	"sync"
	"math"
	// "fmt"
)

type Layer struct {
	name string

	values  []float64
	weights [][]float64
	deltas  []float64

	input  *Layer
	output *Layer

	status    status_
	statusMux sync.Mutex
}

type status_ int8

const (
	// initialized  status_ = iota // 0
	// checkOuts    status_ = iota // 1
	changed      status_ = iota // 2
	evaluated    status_ = iota // 3
	deltas       status_ = iota // 4
	adjusted     status_ = iota // 5
	// weightsAdded status_ = iota // 6
)

func (l *Layer) String() string {
	return "\"" + l.name + "\""
}

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

const bias_value float64 = 1

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

func (l *Layer) getDeltas(targets []float64) error {
	l.statusMux.Lock()
	defer l.statusMux.Unlock()
	if l.status < evaluated {
		return errors.Errorf("Can't get deltas, Layer has not been evaluated")
	} else if l.status >= deltas {
		return nil
	}

	if l.output != nil {
		if err := l.output.getDeltas(targets); err != nil {
			return errors.Wrapf(err, "Can't get deltas, getting deltas of outputs failed\n")
		}

		for v := range l.values {
			var sum float64 // it may be better to just use l.values[v]
			for out := range l.output.deltas {
				sum += l.output.deltas[out] * l.output.weights[out][v]
			}

			l.deltas[v] = sum * l.values[v] * (1 - l.values[v])
		}
	} else {
		if len(targets) != len(l.values) {
			return errors.Errorf("Can't get deltas, len(targets) != len(l.values)")
		}

		for v := range l.values {
			l.deltas[v] = (l.values[v] - targets[v])
		}
	}

	l.status = deltas
	return nil
}

// recurses downwards
func (l *Layer) adjust(learningRate float64) error {
	l.statusMux.Lock()
	if l.status < deltas {
	l.statusMux.Unlock()
		return errors.Errorf("Can't get deltas, layer has not calculated deltas")
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
				// fmt.Println(l, v, w, l.weights[v][w])
				l.weights[v][w] += -1 * learningRate * l.input.values[w] * l.deltas[v]
				// fmt.Println(l, v, w, l.weights[v][w])
			} else {
				// fmt.Println(l, v, w, l.weights[v][w])
				l.weights[v][w] += -1 * learningRate * bias_value * l.deltas[v]
				// fmt.Println(l, v, w, l.weights[v][w])
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
