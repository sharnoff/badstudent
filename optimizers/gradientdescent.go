package optimizers

import (
	"github.com/sharnoff/badstudent"
	"runtime"
)

const threadSizeMultiplier int = 2

type gradientdescent int8

func GradientDescent() gradientdescent {
	return gradientdescent(0)
}

func (g gradientdescent) Run(l *badstudent.Layer, size int, grad func(int) float64, add func(int, float64), learningRate float64) error {

	threadsPerCPU := 1
	opsPerThread := runtime.NumCPU()

	f := func(i int) {
		add(i, -1*learningRate*grad(i))
	}

	badstudent.MultiThread(0, size, f, opsPerThread, threadsPerCPU)

	return nil
}
