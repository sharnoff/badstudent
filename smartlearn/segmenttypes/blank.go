package segmenttypes

import (
	"github.com/sharnoff/smartlearning/smartlearn"
	"github.com/pkg/errors"
)

// blank implements smartlearn.SegmentType
type blank int8

func Blank() blank {
	return blank(0)
}

// sets len(weights) to 0
// sets len(Values) to s.Dims[0]
func (t blank) SetValuesAndWeights(s *Segment) error {
	if len(s.InVals) != 0 {
		return errors.Errorf("Can't SetValuesAndWeights() for segment %s, SegmentType blank should not have inputs (len(s.InVals) == %d)", s.Name, len(s.InVals))
	} else if len(s.Dims) == 0 {
		return errors.Errorf("Can't SetValuesAndWeights() for segment %s, len(s.Dims) == 0", s.Name)
	} else if s.Dims[0] <= 0 {
		return errors.Errorf("Can't SetValuesAndWeights() for segment %s, s.Dims[0] <= 0 (%d). Blank uses dims[0] as the number of values for segment", s.Name, s.Dims[0])
	}

	s.Values = make([]float64, s.Dims[0])
	return nil
}

// assumes SetValuesAndWeights() was already called
func (t blank) EvaluateFunc(s *Segment) (func() error, error) {
	return func() error {
		copy(s.Values, s.InVals)
		return nil
	}, nil
}

// assumes SetValuesAndWeights() was already called
func (t blank) InputDeltasFunc(s *Segment) (func(int, []float64) error, error) {
	return func(int, []float64) error { return nil }, nil
}

// assumes SetValuesAndWeights() was already called
func (t blank) AdjustFunc(s *Segment) (func(float64) error, error) {
	return func(float64) error { return nil }, nil
}