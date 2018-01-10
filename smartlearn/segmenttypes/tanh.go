package segmenttypes

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/smartlearning/smartlearn"
	"math"
)

// @OPTIMIZE : could maybe multithread to make it run faster

// tanh implements smartlearn.SegmentType
type tanh int8

func Tanh() tanh {
	return tanh(0)
}

// sets len(Values) to len(InVals), Dims doesn't matter
// len(weights) is 0
func (t tanh) SetValuesAndWeights(s *smartlearn.Segment) error {
	if len(s.InVals) == 0 {
		return errors.Errorf("Can't SetValuesAndWeights() for segment %s, SegmentType softmax requires inputs (len(s.InVals) == 0)", s.Name)
	}

	s.Values = make([]float64, len(s.InVals))
	return nil
}

func (t tanh) EvaluateFunc(s *smartlearn.Segment) (func() error, error) {
	return func() error {
		for i := range s.InVals {
			s.Values[i] = math.Tanh(s.InVals[i])
		}
		return nil
	}, nil
}

func (t tanh) InputDeltasFunc(s *smartlearn.Segment) (func(int, []float64) error, error) {
	return func(input int, d []float64) error {
		if input >= len(s.NumVpI) {
			return errors.Errorf("Can't get input deltas of segment %s, input >= len(s.NumVpI) (%d >= %d)", s.Name, input, len(s.NumVpI))
		} else if len(d) != s.NumVpI[input] {
			return errors.Errorf("Can't get input deltas of segment %s, len(d) != s.NumVpI[input] (%d != %d)", s.Name, len(d), s.NumVpI[input])
		}

		start := 0
		for i := range s.NumVpI {
			if i == input {
				break
			}
			start += s.NumVpI[i]
		}

		for i := start; i < start + s.NumVpI[input]; i++ {
			d[i] = (1 - math.Pow(s.Values[i], 2)) * s.Deltas[i]
		}

		return nil
	}, nil
}

func (t tanh) AdjustFunc(s *smartlearn.Segment) (func(float64) error, error) {
	return func(learningRate float64) error {
		return nil
	}, nil
}
