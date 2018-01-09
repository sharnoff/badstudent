package segmenttypes

import (
	"github.com/sharnoff/smartlearning/smartlearn"
	"github.com/pkg/errors"
	"math/rand"
)

// @OPTIMIZE : anything can be made much faster by either multi-threading or porting to CUDA

// neurons implements smartlearn.SegmentType
type neurons int8

func Neurons() neurons {
	return neurons(0)
}

// sets len(Values) to Dims[0]
// sets len(Weights) to len(InVals) * len(Values)
func (t neurons) SetValuesAndWeights(s *Segment) error {
	if len(s.InVals) == 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, segment must have inputs (len(s.InVals) == 0)", s.Name)
	} else if len(s.Dims) == 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, len(s.Dims) == 0", s.Name)
	} else if s.Dims[0] <= 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, ", s.Name)
	}

	s.Values = make([]float64, s.Dims[0])
	s.Weights = make([]float64, s.Dims[0] * len(s.InVals))
	for i := range s.Weights {
		s.Weights[i] = 1 / float64(len(s.InVals)) * (2 * rand.Float64() - 1)
	}

	return nil
}

// weights are arranged so that weights[n] is likely
// influencing the same value as weights[n+1]
func (t neurons) EvaluateFunc(s *Segment) (func() error, error) {
	return func() error {
		i := 0
		for v := range s.Values {
			s.Values[v] = 0
			for _, inv := range s.InVals {
				s.Values[v] += inv * s.Weights[i]
				i++
			}
		}

		return nil
	}, nil
}

func (t neurons) InputDeltasFunc(s *Segment) (func(int, []float64) error, error) {
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

		// @OPTIMIZE : this could be a good place to make some improvements by multi-threading
		for di := range d {
			var sum float64
			for i := range s.Deltas {
				sum += s.Weights[len(s.InVals) * i + start + di] * s.Deltas[i]
			}
			d[di] = sum
		}

		return nil
	}, nil
}

func (t neurons) AdjustFunc(s *Segment) (func(float64) error, error) {
	return func(learningRate float64) error {

		for v := range s.Deltas {
			for i := range s.InVals {
				s.Weights[v * len(s.InVals) + i] += -1 * learningRate * s.InVals[i] * s.Deltas[v]
			}
		}

		return nil

	}, nil
}
