package operators

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/pkg/errors"

	"github.com/sharnoff/badstudent/utils"
	"runtime"
	"math"
)

type tanh int8

// Tanh returns an Operator that performs an element-wise application of
// the tanh() function.
func Tanh() tanh {
	return tanh(0)
}

func (t tanh) Init(n *bs.Node) error {
	if n.Size != n.NumInputs() {
		return errors.Errorf("Can't initialize tanh operator, does not have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t tanh) Save(n *bs.Node, dirPath string) error {
	return nil
}

func (t tanh) Load(n *bs.Node, dirPath string, aux []interface{}) error {
	return nil
}

func (t tanh) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = math.Tanh(inputs[i])
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t tanh) Value(n *bs.Node, index int) float64 {
	return math.Tanh(n.InputValue(index))
}

// the derivative of tanh(x) is 1 - tanh(x)^2
func (t tanh) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {

	f := func(i int) {
		d := 1 - math.Pow(n.Value(i-start), 2)
		add(i-start, n.Delta(i-start) * d)
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end f, opsPerThread, threadsperCPU)

	return nil
}

func (t tanh) CanBeAdjusted(n *bs.Node) bool {
	return false
}

func (t tanh) NeedsValues(n *bs.Node) bool {
	return true
}

func (t tanh) NeedsInputs(n *bs.Node) bool {
	return false
}

func (t tanh) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t tanh) AddWeights(n *bs.Node) error {
	return nil
}
