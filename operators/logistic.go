package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"

	"math"
	"runtime"
)

const threadSizeMultiplier int = 1

type logistic int8

// Logistic returns an operator that performs an element-wise application of
// the logistic (or sigmoid) function, using math.Tanh().
//
// The logistic function can be written as:
// l(x) = 0.5 + 0.5 * tanh(0.5 * x)
func Logistic() logistic {
	return logistic(0)
}

func (t logistic) TypeString() string {
	return "logistic"
}

func (t logistic) Init(n *bs.Node) error {
	if n.Size() != n.NumInputs() {
		return errors.Errorf("Can't initialize logistic Operator, does not have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t logistic) Save(n *bs.Node, dirPath string) error {
	return nil
}

func (t logistic) Load(dirPath string) error {
	return nil
}

func (t logistic) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = 0.5 + 0.5*math.Tanh(0.5*inputs[i])
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) Value(n *bs.Node, index int) float64 {
	return 0.5 + 0.5*math.Tanh(0.5*n.InputValue(index))
}

// The derivative of the logistic function l(x) is l(x) * (1 - l(x))
func (t logistic) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {

	f := func(i int) {
		add(i-start, n.Delta(i)*n.Value(i)*(1-n.Value(i)))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t logistic) CanBeAdjusted(n *bs.Node) bool {
	return false
}

func (t logistic) NeedsValues(n *bs.Node) bool {
	return true
}

func (t logistic) NeedsInputs(n *bs.Node) bool {
	return false
}

func (t logistic) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t logistic) AddWeights(n *bs.Node) error {
	return nil
}
