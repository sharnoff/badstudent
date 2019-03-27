package optimizers

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

type sgd int8

// SGD returns the gradient descent Optimzier. SGD requires the hyperparameter "learning-rate" > 0,
// and nothing else.
//
// The result of SGD implements badstudent.Optimzier, and is the default default Optimizer at
// startup, until something else is assigned.
func SGD() sgd {
	return sgd(0)
}

func (s sgd) TypeString() string {
	return "SGD"
}

func (s sgd) Run(n *bs.Node, a bs.Adjustable, ch []float64) {
	η := n.HP("learning-rate")

	f := func(i int) {
		ch[i] += -1 * η * a.Grad(n, i)
	}

	// just arbitrary constants
	threadsPerCPU := 1
	opsPerThread := runtime.NumCPU() * 2
	utils.MultiThread(0, len(ch), f, opsPerThread, threadsPerCPU)
}

func (s sgd) Needs() []string {
	return []string{"learning-rate"}
}
