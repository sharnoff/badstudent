package operators

import "github.com/sharnoff/badstudent"

// optimizers should have a way to store their data, if they have any
type Optimizer interface {
	// arguments: target layer, number of weights, gradient of weight at index,
	// add to weight at index, learning rate
	//
	// number of weights can be 0
	// gradient of weights, adding to weights allows panicing
	// adding to weights is not thread-safe for repeated indexes
	Run(*badstudent.Layer, int, func(int) float64, func(int, float64), float64) error
	// Run(l *Layer, size int, grad func(int) float64, add func(int, float64), learningRate float64) error
}