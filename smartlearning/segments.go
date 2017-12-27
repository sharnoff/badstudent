package smartlearning

import (
	"github.com/pkg/errors"
)

func (s *Segment) SetValues(newVals []float64) error {
	s.mux.Lock()
	defer s.mux.Unlock()

	if len(newVals) != len(s.values) {
		return errors.Errorf("can't set values for segment; len(newVals) != len(s.values) (%d != %d)", len(newVals), len(s.values))
	}

	changed := false
	for i, v := range newVals {
		if s.values[i] != v {
			changed = true
			s.values[i] = v
		}
	}

	if changed == true {
		// update all outputs of s with 'inputsChanged'
		var update func(*Segment)
		update = func(s *Segment) {
			if s.calc != inputsChanged {
				s.calc = inputsChanged
				for _, o := range s.outputs {
					update(o)
				}
			}
		}

		update(s)

		s.calc = evaluated // to make sure that these values stay here
	}

	return nil
}

func (s *Segment) Calculate(result chan error) {
	s.mux.Lock()
	defer s.mux.Unlock()

	if s.calc == evaluated {
		result <- nil
		return
	} else if s.inputs == nil {
		s.calc = evaluated
		result <- nil
		return
	}

	results := make([]chan error, len(s.inputs))
	for i, in := range s.inputs {
		results[i] = make(chan error)
		go in.Calculate(results[i])
	}
	// check that nothing err'd and copy the values if it needs to
	ind := 0 // only used if !inputsIdentical
	for i, c := range results {
		err := <-c
		if err != nil {
			result <- errors.Wrapf(err, "tried to calculate input %d (name: %s)\n", i, s.inputs[i].name)
			return
		}

		if !s.inputsIdentical {
			for inv := range s.inputs[i].values {
				s.inVals[ind] = s.inputs[i].values[inv]
				ind++
			}
		}
	}

	err := s.calculate()
	if err != nil {
		result <- err
		return
	}

	s.calc = evaluated
	result <- nil
	return
}

// essentially Calculate(), but doesn't change if adjustments have been made
func (s *Segment) Values() ([]float64, error) {
	if s.calc >= evaluated {
		return s.values, nil
	}

	result := make(chan error)
	s.Calculate(result)
	err := <-result
	if err != nil {
		return nil, errors.Wrap(err, "couldn't calculate segment\n")
	}

	return s.values, nil
}

func (net *Network) CalcDeltas(s *Segment, targetOutputs []float64) error {
	if s.calc >= delta {
		return nil
	}

	if len(targetOutputs) != len(net.outputs) {
		return errors.Errorf("differing number of target outputs to actual outputs : len(targetOutputs) != len(net.outputs) (%d != %d)", len(targetOutputs), len(net.outputs))
	}

	// makes sure that the values are up to date
	if s.calc != evaluated {
		return errors.Errorf("Segments need to be calculated before deltas. \"%s\".calc == %d", s.name, int8(s.calc))
	}

	results := make([]chan error, len(s.outputs))
	newDeltas := make([][]float64, len(s.outputs))
	for i, o := range s.outputs {
		results[i] = make(chan error)
		newDeltas[i] = make([]float64, len(s.deltas))
		go net.GetInputDeltas(o, s, targetOutputs, newDeltas[i], results[i])
	}

	for i, ch := range results {
		err := <-ch
		if err != nil {
			return errors.Wrapf(err, "couldn't get input deltas from segment output %d (name: %s)\n", i, s.outputs[i].name)
		}
	}

	for i := range s.deltas {
		s.deltas[i] = 0
		for _, d := range newDeltas {
			s.deltas[i] += d[i]
		}
	}

	if s.isOutput {
		// figure out which outputs are the values that it should be looking for
		offset := 0
		for _, o := range net.outSegments {
			if o != s {
				offset += len(o.values)
			} else {
				break
			}
		}

		if offset+len(s.deltas) > len(net.outputs) {
			return errors.Errorf("Couldn't find segment in the network's outputs : offset + len(s.values) >= len(net.outputs) (%d + %d >= %d)", offset, len(s.values), len(net.outputs))
		}

		// add to all of the deltas : current value - target value
		for i := range s.deltas {
			//@CONSTANT : assumes that it is optimizing for the squared error function
			//@CHANGE   : allow for other error functions
			s.deltas[i] += s.values[i] - targetOutputs[offset+i]
		}
	}

	s.calc = delta
	return nil
}

// outputs to deltas
func (net *Network) GetInputDeltas(s *Segment, in *Segment, targetOutputs, deltas []float64, result chan error) {
	s.mux.Lock()
	defer s.mux.Unlock()

	err := net.CalcDeltas(s, targetOutputs)
	if err != nil {
		result <- errors.Wrap(err, "Couldn't calculate own deltas\n")
		return
	}

	ind := 0
	for i := range s.inputs {
		if s.inputs[i] == in {
			ind = i
			goto Found
		}
	}
	result <- errors.New("Couldn't find given input in inputs of segment")
	return

Found:
	err = s.inputDeltas(ind, deltas)
	result <- err
	return
}

func (net *Network) GenerateDeltas(targetOutputs []float64) error {
	results := make([]chan error, len(net.outSegments))
	for i, o := range net.outSegments {
		results[i] = make(chan error)
		go o.Calculate(results[i])
	}
	for i, ch := range results {
		err := <-ch
		if err != nil {
			return errors.Wrapf(err, "couldn't calculate net output %d - \"%s\"\n", i, net.outSegments[i].name)
		}
	}

	for i, in := range net.inSegments {
		err := net.CalcDeltas(in, targetOutputs)
		if err != nil {
			return errors.Wrapf(err, "error from input %d\n", i)
		}
	}

	return nil
}

func (s *Segment) Adjust(learningRate float64, recurseDown bool, result chan error) {
	s.mux.Lock()
	defer s.mux.Unlock()

	if s.calc < delta {
		result <- errors.New("can't adjust because deltas haven't been calculated yet")
		return
	}

	if s.calc < weightsChanged {
		err := s.adjust(learningRate)
		if err != nil {
			result <- err
			return
		}
	}

	if recurseDown {
		results := make([]chan error, len(s.inputs))
		for i, in := range s.inputs {
			results[i] = make(chan error)
			go in.Adjust(learningRate, recurseDown, results[i])
		}

		for i, ch := range results {
			err := <-ch
			if err != nil {
				result <- errors.Wrapf(err, "tried to adjust input %d with downward recursion (name: %s)\n", i, s.inputs[i].name)
				return
			}
		}
	}

	result <- nil
	return
}