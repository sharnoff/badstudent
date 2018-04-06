package operators

import (
	"github.com/sharnoff/smartlearning/badstudent"
	"github.com/pkg/errors"
	"math"
)

type logistic int8

func Logistic() logistic {
	return logistic(0)
}

func (t logistic) Init(l *badstudent.Layer) (error) {
	if l.Size() != l.NumInputs() {
		return errors.Errorf("Can't initialize logistic Operator")
	}

	return nil
}

func (t logistic) Evaluate(l *badstudent.Layer, values []float64) error {
	i := 0
	for in := range l.InputIterator() {
		values[i] = 0.5 + 0.5 * math.Tanh(0.5 * in)
		i++
	}

	return nil
}

func (t logistic) InputDeltas(l *badstudent.Layer, add func(int, float64), input int) error {
	start := l.PreviousInputs(input)
	end := start + l.InputSize(input)

	for in := start; in < end; in++ {
		add(in - start, l.Delta(in) * l.Value(in) * (1 - l.Value(in)))
	}

	return nil
}

func (t logistic) Adjust(l *badstudent.Layer, opt badstudent.Optimizer, learningRate float64) error {
	return nil
}