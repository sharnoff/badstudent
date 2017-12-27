package smartlearning

import ()

// calcSteps are used in order to keep track of which elements of a neuron have been calculated, and which have not
// this prevents unsafe operations, such as calculating how much the weights need to change when the value hasn't been calculated
type calcStep int8

const (
	// nalloc, alloc calcStep = -3, -2, used in *Network.allocateMemory()
	// validOutputs calcStep = -1, used in *Network.SetOutputs()
	inputsChanged  calcStep = 0
	evaluated      calcStep = 1
	delta          calcStep = 2
	weightsChanged calcStep = 3
)

// these aren't really being used much
func (net *Network) setAllCalc(c calcStep, condition, recurse func(calcStep) bool) {
	for _, in := range net.inSegments {
		in.setAllCalcAbove(c, condition, recurse)
	}
}

func (s *Segment) setAllCalcAbove(c calcStep, condition, recurse func(calcStep) bool) {
	r := recurse(s.calc)

	if condition(s.calc) {
		s.calc = c
	}

	if r {
		for _, out := range s.outputs {
			out.setAllCalcAbove(c, condition, recurse)
		}
	}
}
