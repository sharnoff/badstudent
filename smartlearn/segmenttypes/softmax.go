package segmenttypes

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/smartlearning/smartlearn"
	"math"
)

// @OPTIMIZE : could maybe multithread to make it run faster, forking calculation

// softmax implements smartlearn.SegmentType
type softmax int8

func Softmax() softmax {
	return softmax(0)
}

// sets len(Values) to len(InVals), Dims doesn't matter
// len(weights) is 0
func (t softmax) SetValuesAndWeights(s *smartlearn.Segment) error {
	if len(s.InVals) == 0 {
		return errors.Errorf("Can't SetValuesAndWeights() for segment %s, SegmentType softmax requires inputs (len(s.InVals) == 0)", s.Name)
	}

	s.Values = make([]float64, len(s.InVals))
	return nil
}

func (t softmax) EvaluateFunc(s *smartlearn.Segment) (func() error, error) {
	return func() error {
		var sum float64
		for i := range s.Values {
			s.Values[i] = math.Exp(s.InVals[i])
			sum += s.Values[i]
		}

		for i := range s.Values {
			s.Values[i] /= sum
		}

		return nil
	}, nil
}

func (t softmax) InputDeltasFunc(s *smartlearn.Segment) (func(int, []float64) error, error) {
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

		for i := range d {
			d[i] = s.Values[i] * (1 - s.Values[i])
		}

		return nil
	}, nil
}

func (t softmax) AdjustFunc(s *smartlearn.Segment) (func(float64) error, error) {
	return func(learningRate float64) error {
		return nil
	}, nil
}
