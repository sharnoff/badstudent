package optimizers

import (
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"runtime"
)

const threadSizeMultiplier int = 2

type gradientdescent int8

func GradientDescent() gradientdescent {
	return gradientdescent(0)
}

func (g gradientdescent) Run(n *badstudent.Node, size int, grad func(int) float64, add func(int, float64), learningRate float64) error {

	threadsPerCPU := 1
	opsPerThread := runtime.NumCPU()

	f := func(i int) {
		add(i, -1*learningRate*grad(i))
	}

	utils.MultiThread(0, size, f, opsPerThread, threadsPerCPU)

	return nil
}

func (g gradientdescent) Save(n *badstudent.Node, op badstudent.Operator, dirPath string) error {
	return nil
}

func (g gradientdescent) Load(n *badstudent.Node, op badstudent.Operator, dirPath string, aux []interface{}) error {
	return nil
}
