package optimizers

import (
	"github.com/sharnoff/badstudent"
)

type gradientdescent int8

func GradientDescent() gradientdescent {
	return gradientdescent(0)
}

func (g gradientdescent) Run(l *badstudent.Layer, size int, grad func(int) float64, add func(int, float64), learningRate float64) error {

	for i := 0; i < size; i++ {
		add(i, -1 * learningRate * grad(i))
	}

	return nil
}
