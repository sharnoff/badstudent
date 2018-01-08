package smartlearning

import ()

type SegmentType interface {
	// this should just make an empty slice for values and initialize all of the weights
	SetValuesAndWeights(*Segment) (error)

	EvaluateFunc(*Segment) (func() error, error)
	InputDeltasFunc(*Segment) (func(int, []float64) error, error)
	AdjustFunc(*Segment) (func(float64) error, error)
}