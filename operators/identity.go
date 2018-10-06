package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"

	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

type identity int8

// Identity returns an operator that is its inputs. This type of operator
// could be useful for delaying one output of a Node but not another,
// such as in LSTMs
func Identity() identity {
	return identity(0)
}

func (t identity) TypeString() string {
	return "identity"
}

func (t identity) Init(n *bs.Node) error {
	if n.Size() != n.NumInputs() {
		return errors.Errorf("Can't initialize identity operator, does not have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t identity) Save(n *bs.Node, dirPath string) error {
	return nil
}

func (t identity) Load(n *bs.Node, dirPath string) error {
	return nil
}

func (t identity) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = inputs[i]
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t identity) Value(n *bs.Node, index int) float64 {
	return n.InputValue(index)
}

func (t identity) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {

	f := func(i int) {
		add(i-start, n.Delta(i-start))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t identity) CanBeAdjusted(n *bs.Node) bool {
	return false
}

func (t identity) NeedsValues(n *bs.Node) bool {
	return false
}

func (t identity) NeedsInputs(n *bs.Node) bool {
	return false
}

func (t identity) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t identity) AddWeights(n *bs.Node) error {
	return nil
}
