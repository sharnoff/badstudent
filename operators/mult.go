package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

type mult int8

// Multiplies its inputs. All inputs must have the same size as node.
// Can loop to make forget gate of LSTM
func Mult() mult {
	return mult(0)
}

func (t mult) TypeString() string {
	return "multiply"
}

func (t mult) Init(n *bs.Node) error {
	if n.NumInputs() == 0 {
		return errors.Errorf("Must have >= 1 input")
	}

	for i := 0; i < n.NumInputNodes(); i++ {
		if n.InputSize(i) != n.Size() {
			return errors.Errorf("Size of input %d is not equal to node (%d != %d)", i, n.InputSize(i), n.Size())
		}
	}

	return nil
}

// does not save anything
func (t mult) Save(n *bs.Node, dirPath string) error {
	return nil
}

// does not save anything
func (t mult) Load(dirPath string) error {
	return nil
}

func (t mult) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	f := func(i int) {
		values[i] = inputs[i]
		for in := 1; in < n.NumInputNodes(); in++ {
			values[i] *= inputs[in*n.Size()+i]
		}
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)

	return nil
}

func (t mult) Value(n *bs.Node, index int) float64 {

	v := n.InputValue(index)
	for in := 1; in < n.NumInputNodes(); in++ {
		v *= n.InputValue(in*n.Size() + index)
	}

	return v
}

func (t mult) InputDeltas(n *bs.Node, mult func(int, float64), start, end int) error {
	f := func(i int) {
		mult(i-start, n.Delta(i%n.Size())*n.InputValue(i))
	}

	opsPerThread := runtime.NumCPU() * threadSizeMultiplier
	threadsPerCPU := 1

	utils.MultiThread(start, end, f, opsPerThread, threadsPerCPU)

	return nil
}

func (t mult) CanBeAdjusted(n *bs.Node) bool {
	return false
}

func (t mult) NeedsValues(n *bs.Node) bool {
	return false
}

func (t mult) NeedsInputs(n *bs.Node) bool {
	return true
}

func (t mult) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (t mult) AddWeights(n *bs.Node) error {
	return nil
}
