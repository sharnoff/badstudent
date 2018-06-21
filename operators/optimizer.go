package operators

import "github.com/sharnoff/badstudent"

// The way in which the weights of any Operator provided here is adjusted
type Optimizer interface {
	// arguments: target layer, number of weights, gradient of weight at index,
	// add to weight at index, learning rate
	//
	// number of weights can be 0
	// gradient of weights, adding to weights allows panicing
	// adding to weights is not thread-safe for repeated indexes
	//
	// Run(l *Layer, size int, grad func(int) float64, add func(int, float64), learningRate float64) error
	Run(*badstudent.Layer, int, func(int) float64, func(int, float64), float64) error

	// given a path to the directory (and the name of it, ending without '/'),
	// should store enough information to recreate the Optimizer from file
	//
	// the directory will not be created, used, or altered by anything else
	Save(*badstudent.Layer, badstudent.Operator, string) error

	// given a path to the directory (and the name of it, ending without '/'),
	// should recreate the Optimizer with the information given
	//
	// the directory will not be created, used, or altered by anything else
	Load(*badstudent.Layer, badstudent.Operator, string, []interface{}) error
}
