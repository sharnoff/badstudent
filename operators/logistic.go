package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"math"
	"runtime"
)

const threadSizeMultiplier int = 1

type logistic int8

func Logistic() logistic {
	return logistic(0)
}

func (t logistic) Init(l *badstudent.Layer) error {
	if l.Size() != l.NumInputs() {
		return errors.Errorf("Can't initialize logistic Operator, does not have same number of values as inputs")
	}

	return nil
}

func (t logistic) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()

	f := func(i int) {
		values[i] = 0.5 + 0.5*math.Tanh(0.5*inputs[i])
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	badstudent.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

	f := func(i int) {
		add(i-start, l.Delta(i)*l.Value(i)*(1-l.Value(i)))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	badstudent.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) CanBeAdjusted(l *badstudent.Layer) bool {
	return false
}

func (t logistic) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	return nil
}

func (t logistic) AddWeights(l *badstudent.Layer) error {
	return nil
}
