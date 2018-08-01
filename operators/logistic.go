package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"math"
	"runtime"
)

const threadSizeMultiplier int = 1

type logistic int8

func Logistic() logistic {
	return logistic(0)
}

func (t logistic) Init(n *badstudent.Node) error {
	if n.Size() != n.NumInputs() {
		return errors.Errorf("Can't initialize logistic Operator, does not have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t logistic) Save(n *badstudent.Node, dirPath string) error {
	return nil
}

func (t logistic) Load(n *badstudent.Node, dirPath string, aux []interface{}) error {
	if n.Size() != n.NumInputs() {
		return errors.Errorf("Can't load logistic Operator, does not have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t logistic) Evaluate(n *badstudent.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = 0.5 + 0.5*math.Tanh(0.5*inputs[i])
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) InputDeltas(n *badstudent.Node, add func(int, float64), start, end int) error {

	f := func(i int) {
		add(i-start, n.Delta(i) * n.Value(i) * (1 - n.Value(i)))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) CanBeAdjusted(n *badstudent.Node) bool {
	return false
}

func (t logistic) Adjust(n *badstudent.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t logistic) AddWeights(n *badstudent.Node) error {
	return nil
}
