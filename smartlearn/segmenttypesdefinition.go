package smartlearn

import ()

type SegmentType interface {
	// this should just make an empty slice for values and initialize all of the weights
	// the segment members that have been initialized at this point are:
	//		- Name
	//		- Dims
	//		- NumVpI
	//		- InputDims
	//		- InVals
	SetValuesAndWeights(*Segment) (error)

	EvaluateFunc(*Segment) (func() error, error)
	InputDeltasFunc(*Segment) (func(int, []float64) error, error)
	AdjustFunc(*Segment) (func(float64) error, error)
}

// useful signatures to copy and paste:
// func (t type) SetValuesAndWeights(s *Segment) error {}
// func (t type) EvaluateFunc(s *Segment) (func() error, error) {}
// func (t type) InputDeltasFunc(s *Segment) (func(int, []float64) error, error) {}
// func (t type) AdjustFunc(s *Segment) (func(float64) error, error) {}
