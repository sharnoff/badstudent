package smartlearning

import (
	errs "FromGithub/errors"
	"fmt"
)

type Segment struct {
	inputs    []*Segment
	outputs   []*Segment
	values    []float64
	inValues  []float64
	weights   []float64
	dims      []int
	inputDims [][]int
	isOutput  bool

	// @CHANGE : it is probably just better to include the SegmentType, instead of these redundant functions
	calc calcStep
	// "this" : error
	calculate func(*Segment) error
	deltas    []float64 // how much the total error changes if these values change
	// "this", target segment, output segments, target values for those outputs : deltas, error
	// inputDeltas func(*Segment, *Segment, []*Segment, [][]float64) ([]float64, error)
	// "this", target segment : deltas, error
	inputDeltas func(*Segment, *Segment) ([]float64, error) // @OPTIMIZE : could have these output to given slice using channels
	// "this", learning rate : error
	adjust func(*Segment, float64) error

	name string
}

type Network struct {
	inputs  []*Segment
	outputs []*Segment

	numInputs  int
	numOutputs int
}

func New(numInputs, numOutputs int) (*Network, error) {
	if numInputs <= 0 {
		return nil, errs.Errorf("numInputs <= 0 (%d)", numInputs)
	} else if numOutputs <= 0 {
		return nil, errs.Errorf("numOutputs <= 0 (%d)", numOutputs)
	}

	net := new(Network)

	net.numInputs = numInputs
	net.numOutputs = numOutputs
	net.inputs = make([]*Segment, 0)
	// net.outputs = nil

	return net, nil
}

func (net *Network) NewSegment(name string, typ *SegmentType, dims []int, inputs ...*Segment) (*Segment, error) {
	if typ == nil {
		return nil, errs.Errorf("typ == nil")
	} else if net.outputs != nil {
		return nil, errs.Errorf("Network has already been finished with its construction (net.outputs != nil)")
	}

	inputDims := make([][]int, len(inputs))
	numInputValues := make([]int, len(inputs))
	totalNumInputs := 0
	for i := range inputs {
		inputDims[i] = inputs[i].dims
		numInputValues[i] = len(inputs[i].values)
		totalNumInputs += numInputValues[i]
	}

	numValues, numWeights, err := typ.NumValuesAndWeights(inputDims, numInputValues, dims)
	if err != nil {
		return nil, errs.Wrap(err, "typ.NumValuesANdWeights() returned non-nil error")
	}

	seg := new(Segment)
	seg.inputs = inputs
	// seg.outputs = nil, used later to detect for useless segments
	seg.values = make([]float64, numValues)
	seg.inValues = make([]float64, totalNumInputs) // @OPTIMIZE : could arrange this so that inValues points to the values of the inputs
	seg.weights = make([]float64, numWeights)
	seg.dims = dims
	seg.inputDims = inputDims

	// @CHANGE : it would be better to just have the segment methods do this, instead of right here
	seg.calc = inputsChanged
	seg.calculate = func(s *Segment) error {
		return typ.Calculate(s.inputDims, s.inValues, s.dims, s.weights, s.values)
	}
	seg.deltas = make([]float64, numValues)
	seg.inputDeltas = func(s *Segment, in *Segment) ([]float64, error) {
		return typ.InputDeltas(s.inputs, in, s.inputDims, s.inValues, s.dims, s.weights, s.deltas)
	}
	seg.adjust = func(s *Segment, learningRate float64) error {
		return typ.Adjust(s.weights, s.deltas, learningRate)
	}

	seg.name = name

	for _, in := range inputs {
		in.outputs = append(in.outputs, seg)
	}

	return seg, nil
}

func (net *Network) SetOutputs(outputs ...*Segment) error {
	if net.outputs != nil {
		return errs.Errorf("net.outputs != nil (%v)", net.outputs)
	} else if len(outputs) <= 0 {
		return errs.Errorf("len(outputs) <= 0 (%d)", len(outputs))
	}

	net.outputs = outputs
	for _, o := range outputs {
		o.isOutput = true
	}

	// check that none of the segments have no outputs and aren't one of the output segments
	// this would be a problem, because the
	var validOutputs calcStep = -1

	var scanOutputs func(*Segment) error
	scanOutputs = func(s *Segment) error {
		if s.calc == validOutputs {
			return nil
		}

		if len(s.outputs) == 0 /*|| s.outputs == nil */ {
			if s.isOutput == false {
				return errs.Errorf("segment %s's outputs do not affect network outputs (s.isOutput == false)", s.name)
			}
			return nil
		} else {
			for i, o := range s.outputs { //@OPTIMIZE: could be multi-threaded, but not really necessary
				err := scanOutputs(o)
				if err != nil {
					return errs.Wrap(err, fmt.Sprintf("at output %d", i))
				}
			}
		}

		s.calc = validOutputs
		return nil
	}

	// every segment is guaranteed to be tracible to the inputs, so we start from there, checking that there are no segments that can't be traced to the outputs
	for i, in := range net.inputs {
		err := scanOutputs(in)
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("from input %d", i))
		}
	}

	return nil
}
