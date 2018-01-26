package segmenttypes

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/smartlearning/smartlearn"
	"math/rand"
	"math"
	// "fmt"
)

// @OPTIMIZE : anything can be made much faster by either multi-threading or porting to CUDA

// neurons implements smartlearn.SegmentType
type neurons int8

func Neurons() neurons {
	return neurons(0)
}

const bias_value float64 = 1.0

// sets len(Values) to Dims[0]
// sets len(Weights) to len(InVals) * (len(Values) + 1) (for bias)
func (t neurons) SetValuesAndWeights(s *smartlearn.Segment) error {
	if len(s.InVals) == 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, segment must have inputs (len(s.InVals) == 0)", s.Name)
	} else if len(s.Dims) == 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, len(s.Dims) == 0", s.Name)
	} else if s.Dims[0] <= 0 {
		return errors.Errorf("Couldn't SetValuesAndWeights() for segment %s, ", s.Name)
	}

	s.Values = make([]float64, s.Dims[0])
	s.Weights = make([]float64, s.Dims[0] * (len(s.InVals) + 1))
	for i := range s.Weights {
		s.Weights[i] = 1 / float64(len(s.InVals) + 1) * (2*rand.Float64() - 1)
	}

	return nil
}

// weights are arranged so that weights[n] is likely
// influencing the same value as weights[n+1],
// and taking input from inVals[n%len(inVals)]
// biases are appended to the end of the section of weights that a value takes input from
func (t neurons) EvaluateFunc(s *smartlearn.Segment) (func() error, error) {
	return func() error {
		for v := range s.Values {
			w := s.Weights[v * (len(s.InVals) + 1) : (v + 1) * (len(s.InVals) + 1)]

			s.Values[v] = 0

			for i, inv := range s.InVals {
				s.Values[v] += inv * w[i]
			}
			s.Values[v] += bias_value * w[len(w) - 1]

			if math.IsNaN(s.Values[v]) {
				return errors.Errorf("Couldn't evaluate segment %s, s.Values[%d] results as NaN\n", s.Name, v)
			}
		}

		return nil
	}, nil
}

func (t neurons) InputDeltasFunc(s *smartlearn.Segment) (func(int, []float64) error, error) {
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

		for di := range d {
			var sum float64
			for i := range s.Deltas {
				sum += s.Weights[start + i*(len(s.InVals) + 1) + di] * s.Deltas[i]
			}
			d[di] = sum

			if math.IsNaN(sum) {
				return errors.Errorf("Can't get input deltas of semgent %s, d[%d] resulted evaluated to NaN", s.Name, di)
			}
		}

		return nil
	}, nil
}

func (t neurons) AdjustFunc(s *smartlearn.Segment) (func(float64) error, error) {
	return func(learningRate float64) error {

		for v := range s.Deltas {
			w := s.Weights[v * (len(s.InVals) + 1) : (v + 1) * (len(s.InVals) + 1)]
			for i := range s.InVals {
				w[i] -= learningRate * s.InVals[i] * s.Deltas[v]
			}

			// biases
			w[len(w) - 1] -= learningRate * bias_value * s.Deltas[v]
		}

		return nil

	}, nil
}
