package smartlearning

import (
	errs "FromGithub/errors"
	"fmt"
)

func (s *Segment) SetValues(newVals []float64) error {
	if len(newVals) != len(s.values) {
		return errs.Errorf("len(newVals) != len(s.values)")
	}

	changed := false
	for i, v := range newVals {
		if s.values[i] != v {
			changed = true
			break
		}
	}
	s.values = newVals

	if changed == true {
		s.calc = evaluated

		// update all of the outputs to s with 'inputsChanged'
		var update func(s *Segment)
		update = func(s *Segment) {
			s.calc = inputsChanged
			for _, o := range s.outputs {
				update(o)
			}
		}

		update(s)
	}

	return nil
}

func (s *Segment) Calculate() error {
	if s.inputs == nil || s.calc >= evaluated {
		return nil
	}

	ivi := 0
	for i, in := range s.inputs { // @OPTIMIZE : definitely needs to be multi-threaded... use sync.Mutex locks
		err := in.Calculate()
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("from input %d", i))
		}

		for iv := range in.values { // @OPTIMIZE : should only update inValues if in's values have changed
			s.inValues[ivi] = in.values[iv]
			ivi++
		}
	}

	// calculate isn't a method, it's a member of the structure
	err := s.calculate(s)
	if err != nil {
		return err
	}

	s.calc = evaluated
	return nil
}

// essentially Calculate(), but doesn't change if adjustments have been made
func (s *Segment) Values() ([]float64, error) {
	if s.calc < evaluated {
		err := s.Calculate()
		return s.values, err
	}
	return s.values, nil
}

func (s *Segment) CalcDeltas(net *Network, targetOutputs []float64) error {
	if s.calc >= delta {
		return nil
	}

	if len(targetOutputs) != net.numOutputs {
		return errs.Errorf("differing number of target outputs to actual outputs : len(targetOutputs) != net.numOutputs (%d != %d)", len(targetOutputs), net.numOutputs)
	}

	err := s.Calculate()
	if err != nil {
		return errs.Wrap(err, "tried to update value")
	}

	addTo := func(a, b []float64) error {
		if len(a) != len(b) {
			return errs.Errorf("can't add: len(a) != len(b) (%d != %d)", len(a), len(b))
		}

		for i := range a {
			a[i] += b[i]
		}
		
		return nil
	}

	s.deltas = make([]float64, len(s.deltas)) // set all of the values to 0

	for i, o := range s.outputs {
		deltaFromHigher, err := o.GetInputDeltas(net, s, targetOutputs)
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("couldn't get deltas from output %d", i))
		}
		err = addTo(s.deltas, deltaFromHigher)
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("couldn't add to s.deltas from output %d", i))
		}
	}

	// if it's an output segment, also add the deltas from that
	if s.isOutput {
		// figure out which outputs are the values that it should be looking for
		offset := 0
		for _, o := range net.outputs {
			if o != s {
				offset += len(o.values)
			} else {
				break
			}
		}

		if offset+len(s.deltas) >= net.numOutputs {
			return errs.Errorf("Couldn't find segment in the network's outputs : offset + len(s.values) >= net.numOutputs (%d + %d >= %d)", offset, len(s.values), net.numOutputs)
		}

		// add to all of the deltas : current value - target value
		for i := range s.deltas {
			//@CONSTANT : assumes that it is optimizing for the squared error function 
			//@CHANGE   : allow for other error functions
			s.deltas[i] += s.values[i] - targetOutputs[offset+i]
		}
	}

	return nil
}

// @CHANGE : could change this so that it outputs a channel, so that the values can be processed with greater ease
func (s *Segment) GetInputDeltas(net *Network, in *Segment, targetOutputs []float64) ([]float64, error) {
	err := s.CalcDeltas(net, targetOutputs)
	if err != nil {
		return nil, errs.Wrap(err, "tried to calculate s.deltas")
	}

	return s.inputDeltas(s, in)
}

func (net *Network) GenerateDeltas(targetOutputs []float64) error {
	for i, in := range net.inputs {
		err := in.CalcDeltas(net, targetOutputs)
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("error from input %d", i))
		}
	}

	return nil
}

func (s *Segment) Adjust(learningRate float64, recurseDown bool) error {
	if s.calc < delta {
		return errs.New("can't adjust because deltas haven't been calculated yet")
	}

	if s.calc < weightsChanged {
		err := s.adjust(s, learningRate)
		if err != nil {
			return err
		}
	}

	if recurseDown {
		// @OPTIMIZE: needs to be multi-threaded, similar to (*Segment).Calculate
		for i, in := range s.inputs {
			err := in.Adjust(learningRate, recurseDown)
			if err != nil {
				return errs.Wrap(err, fmt.Sprintf("tried to adjust input %d with downward recursion", i))
			}
		}
	}

	return nil
}
