package smartlearning

import (
	"github.com/pkg/errors"
	"sync"
)

type Segment struct {
	name string
	calc calcStep
	mux  sync.Mutex

	inputs          []*Segment
	inVals          []float64
	inputDims       [][]int
	numVpI          []int // number of values in each input to this segment
	inputsIdentical bool
	inputDeltas     func(int, []float64) error

	values    []float64
	dims      []int
	calculate func() error
	weights   []float64
	deltas    []float64
	adjust    func(float64) error

	outputs  []*Segment
	isOutput bool

	typ SegmentType
}

type Network struct {
	inputs     []float64
	inSegments []*Segment

	outputs     []float64
	outSegments []*Segment
}

func New(numInputs, numOutputs int) (*Network, error) {
	if numInputs <= 0 {
		return nil, errors.Errorf("numInputs <= 0 (%d)", numInputs)
	} else if numOutputs <= 0 {
		return nil, errors.Errorf("numOutputs <= 0 (%d)", numOutputs)
	}

	net := new(Network)

	net.inputs = make([]float64, numInputs)
	// net.inputs = make([]*Segment, 0)
	net.outputs = make([]float64, numOutputs)
	// net.outputs = nil

	return net, nil
}

// returns start, error for a new segment
func (net *Network) getInputIndex(s *Segment, length int) (int, error) {
	// count up the number of values already taken
	start := 0
	for _, in := range net.inSegments {
		start += len(in.values)
	}

	if start+length > len(net.inputs) {
		return 0, errors.Errorf("start + length > len(net.inputs) (%d + %d > %d)", start, length, len(net.inputs))
	}

	net.inSegments = append(net.inSegments, s)
	return start, nil
}

// @CHANGE : rename numInputs, numInputValues, etc. to have better names
// numInputs -> total number of inputs
// numVpI -> numInputValues -> num values per each segment
// this is NewSegment()
func (net *Network) Add(name string, typ SegmentType, dims []int, inputs ...*Segment) (*Segment, error) {
	if net.outSegments != nil {
		return nil, errors.Errorf("Can't add segment \"%s\", network outputs have already been set (net.outSegments != nil)", name)
	} else if typ == nil {
		return nil, errors.Errorf("Can't add segment \"%s\", typ SegmentType == nil", name)
	}

	inputDims := make([][]int, len(inputs)) // dimensions of the inputs
	numVpI := make([]int, len(inputs)) // number of values in each input to segment
	numInputs := 0  // total number of inputs to segment
	for i := range inputs {
		inputDims[i] = inputs[i].dims
		numVpI[i] = len(inputs[i].values)
		numInputs += numVpI[i]
	}

	numValues, numWeights, err := typ.NumValuesAndWeights(inputDims, numVpI, numInputs, dims)
	if err != nil {
		return nil, errors.Wrapf(err, "Can't add segment \"%s\", couldn't to get num values/weights", name)
	}

	s := new(Segment)

	if len(inputs) <= 0 { // if it's an input segment
		net.inSegments = append(net.inSegments, s)
		// shouldn't need to append to net.inputs - @WARNING : could be the cause of a bug
		s.inputsIdentical = true
	} else { // if it's not an input segment
		s.inputs = inputs
		s.inVals = make([]float64, 0, numInputs)
		s.inputDims = inputDims
		s.numVpI = numVpI
		// s.inputsIdentical, inputDeltas will be done later in SetOutputs()

		for _, in := range s.inputs {
			in.outputs = append(in.outputs, s)
		}
	}

	s.name = name
	s.calc = inputsChanged
	s.typ = typ
	s.dims = dims
	s.values = make([]float64, numValues) // just a placeholder, it's actually set later in allocateMemory()
	s.deltas = make([]float64, numValues)
	s.weights = make([]float64, numWeights)
	err = typ.InitWeights(s.weights, inputDims, numVpI, numInputs, dims)
	if err != nil {
		return nil, errors.Wrapf(err, "Can't add segment \"%s\", typ.InitWeights() failed", name)
	}

	// s.calculate, adjust will be done later in SetOutputs()

	return s, nil
}

func (net *Network) SetOutputs(outputs ...*Segment) error {
	if net.outSegments != nil {
		return errors.New("Network has already been finalized")
	} else if len(outputs) <= 0 {
		return errors.Errorf("Can't finalize network with <= 0 outputs (%d)", len(outputs))
	}

	// mark all of the outputs as outputs
	net.outSegments = outputs
	for _, o := range outputs {
		if o.isOutput == true {
			return errors.New("Network was/is already partway through being initialized")
		}
		o.isOutput = true
	}

	// search the network to see if there are any segments that don't affect the outputs
	// this would be true if there is a segment that has no outputs but isn't a network output
	{
		var validOutputs calcStep = -1

		var scanOutputs func(*Segment) error
		scanOutputs = func(s *Segment) error {
			if s.calc == validOutputs {
				return nil
			}

			if len(s.outputs) == 0 /*|| s.outputs == nil */ {
				if s.isOutput == false {
					return errors.Errorf("segment %s's outputs do not affect network outputs (s.isOutput == false && len(s.outputs) == 0)", s.name)
				}
				return nil
			} else {
				for i, o := range s.outputs { //@OPTIMIZE: could be multi-threaded, but not really necessary
					err := scanOutputs(o)
					if err != nil {
						return errors.Wrapf(err, "at output %d\n", i)
					}
				}
			}

			s.calc = validOutputs
			return nil
		}

		// every segment is guaranteed to be tracible to the inputs, so we start from there
		for i, in := range net.inSegments {
			err := scanOutputs(in)
			if err != nil {
				return errors.Wrapf(err, "from input %d\n", i)
			}
		}
	}

	// check that none of the inputs are also outputs
	for i, in := range net.inSegments {
		if in.isOutput {
			return errors.Errorf("Segment %s (input %d) is both input and output", in.name, i)
		}
	}

	// allocate the semi-optimal shared memory for each of the segments, starting with the outputs
	err := net.allocateMemory()
	if err != nil {
		return errors.Wrap(err, "Couldn't allocate shared memory for the network\n")
	}

	// now that the memory has been allocated, we can set the 'methods' for each segment
	var setMethods func(*Segment) error
	setMethods = func(s *Segment) (err error) {
		// uses s.inputDeltas == nil to tell if this function has already run
		if s.inputDeltas != nil {
			return nil
		}

		s.inputDeltas, err = s.typ.InputDeltasFunc(s.numVpI, s.inputDims, s.inVals, s.weights, s.values, s.deltas)
		if err != nil {
			return errors.Wrap(err, "couldn't set s.inputDeltas()\n")
		} else if s.inputDeltas == nil {
			return errors.New("s.typ.getInputDeltasFunc() returned nil function, but no error")
		}

		s.calculate, err = s.typ.CalculateFunc(s.inVals, s.weights, s.inputDims, s.dims, s.values)
		if err != nil {
			return errors.Wrapf(err, "couldn't set s.calculate()\n")
		} else if s.calculate == nil {
			return errors.New("s.typ.getCalculateFunc() returned nil function, but no error")
		}

		s.adjust, err = s.typ.AdjustFunc(s.deltas, s.weights, s.inVals)
		if err != nil {
			return errors.Wrap(err, "couldn't set s.adjust()\n")
		} else if s.adjust == nil {
			return errors.New("s.typ.getAdjustFunc() returned nil function, but no error")
		}

		for i, o := range s.outputs {
			err = setMethods(o)
			if err != nil {
				return errors.Wrapf(err, "couldn't set methods for output %d (name: %s)\n", i, o.name)
			}
		}

		return
	}

	for i, in := range net.inSegments {
		err := setMethods(in)
		if err != nil {
			return errors.Wrapf(err, "couldn't set methods for net input %d (name: %s)\n", i, in.name)
		}
	}

	return nil
}

// just a greedy algorithm to allocate the (mostly) shared memory for values
// what this function ensures:
// - all input segments will have their values physically located so that the network can iterate over them
// - the same thing for outputs
// - the same thing for MOST non-i/o segments
func (net *Network) allocateMemory() error {
	// because this is only ever run in *Network.SetOutputs(),
	// we know that all of the calcSteps should be validOutputs, or -1

	var nalloc, allocated calcStep = -3, -2
	memBlocks := make([][]*Segment, 2) // 1 for inputs, 1 for outputs

	// figure out which segments should be put together, in two parts:
	// first, mark all segments with "nalloc", meaning that they have not been put into a memBlock
	{
		condition := func(c calcStep) bool { return true }
		recurse := func(c calcStep) bool { return c != nalloc }
		net.setAllCalc(nalloc, condition, recurse)
	}

	// then, put all of the inputs and outputs into separate memBlocks
	{
		// we know that this is a safe operation, because we check in
		// *Network.SetOutputs() that none of the inputs are also outputs
		for io := 0; io <= 1; io++ {
			var segs []*Segment
			if io == 0 {
				segs = net.inSegments
			} else {
				segs = net.outSegments
			}

			memBlocks[io] = make([]*Segment, len(segs))
			for i, s := range segs {
				memBlocks[io][i] = s
				s.calc = allocated

				for _, out := range s.outputs {
					if out.inputs[0] != s || len(out.inputs)+i >= len(memBlocks[0]) {
						continue
					}

					// check that it's actually the same
					for ind, in := range out.inputs {
						if memBlocks[io][i+ind] != in {
							goto Continue
						}
					}
					out.inputsIdentical = true

				Continue:
				}
			}
		}
	}

	// then, put the rest of the segments into their own memBlocks,
	// based around the initial constraints of the inputs and outputs

	// alloc is designed to attempt to put a segment's inputs into their own memBlock,
	// not allocate itself
	var alloc func(*Segment) error  // alloc isn't quite the best name for this,
	alloc = func(s *Segment) error { // but it's sufficient
		if s.inputs == nil {
			return nil
		}

		// no need to do anything if all of the inputs have already been put into memBlocks
		if s.inputsIdentical {
			return nil
		}
		var indNotAlloc []int
		for _, in := range s.inputs {
			if in.calc == nalloc {
				indNotAlloc = append(indNotAlloc)
			}
		}
		if len(indNotAlloc) == 0 {
			return nil
		}

		// 'goto Return' is used frequently here in order to ensure that
		// the inputs still get allocated
		// this is what it does:
		// Return:
		// for all inputs not allocated
		//     allocate
		//     return err if generated
		// return nil

		// if none of the inputs have been allocated, then allocate them all
		if len(indNotAlloc) == len(s.inputs) {
			mb := s.inputs
			memBlocks = append(memBlocks, mb)

			for _, in := range s.inputs {
				in.calc = allocated
			}

			s.inputsIdentical = true

			goto Return
		} else {
			// check whether the inputs that aren't alloc can be added to a single memBlock of other inputs
			// this can only happen if there is just one 'gap,'
			gap := []int{-1, -1}
			lastV := -1
			for _, v := range indNotAlloc {
				if v != lastV+1 {
					if gap[1] != -1 {
						gap = []int{lastV + 1, v}
					} else {
						goto Return
					}
				}
				lastV = v
			}
			if lastV != len(s.inputs)-1 {
				if gap[1] != -1 {
					gap = []int{lastV + 1, len(s.inputs)}
				} else {
					goto Return
				}
			}

			// given the gap, try to find the memBlock that contains it, if it's just one
			var mb []*Segment
			var mbI int // mbI is used later
			var ind int
			for bI, b := range memBlocks {
				for i, seg := range b {
					if seg == s.inputs[gap[0]] {
						mb = b
						mbI = bI
						ind = i
						goto Break
					}
				}
			}
			// only here if didn't find memBlock containing start of gap
			return errors.Errorf("Couldn't find memBlock containing allocated segment (name: %s)", s.inputs[gap[0]].name)

		Break:
			// check that mb actually contains the rest of the gap
			for i := 0; i < gap[1]-gap[0]; i++ {
				if mb[ind+i] != s.inputs[gap[0]+i] {
					goto Return
				}
			}

			// check if mb can add the other inputs to its end(s)
			gapStart := (gap[0] != 0)             // at the start of the inputs
			gapEnd := (gap[1] != len(s.inputs)-1) // at the end of the inputs
			if !((!gapStart || ind+gap[1]-gap[0] == len(mb)-1) && (!gapEnd || ind == 0)) {
				goto Return
			}

			// add the inputs
			// mbI is from the for loop where we found mb
			memBlocks[mbI] = append(mb, s.inputs[gap[1]:]...)
			memBlocks[mbI] = append(s.inputs[:gap[0]], mb...)

			s.inputsIdentical = true
			goto Return
		}

	Return:
		for _, i := range indNotAlloc {
			err := alloc(s.inputs[i])
			if err != nil {
				return errors.Wrapf(err, "tried to alloc input %d (name: %s)\n", i, s.inputs[i].name)
			}
		}
		return nil
	}
	for i, out := range net.outSegments {
		err := alloc(out)
		if err != nil {
			return errors.Wrapf(err, "tried to alloc net output %d (name: %s)\n", i, out.name)
		}
	}

	// set up all of the segments that are in memBlocks
	for _, mb := range memBlocks {
		var totalSize int
		for _, s := range mb {
			totalSize += len(s.values)
		}

		input, output := false, false
		inVInd, inInd := 0, 0
		outVInd, outInd := 0, 0
		vals := make([]float64, totalSize)
		dims := make([][]int, len(mb))
		numValues := make([]int, len(mb))
		ind := 0
		for i, s := range mb {
			end := ind + len(s.values)
			s.values = vals[ind:end]
			dims[i] = s.dims
			numValues[i] = len(s.values)

			// also set up inVals for the segments above
			for _, o := range s.outputs {
				if o.inputs[0] == s && len(o.inVals) == 0 && o.inputsIdentical {
					if ind+cap(o.inVals) >= totalSize {
						return errors.Errorf("Number of inVals vs number of input values didn't match. Name: %s, inVals: %d, (remaining) inputs: %d", o.name, cap(o.inVals), totalSize-ind)
					} else if i+len(o.inputs) >= len(mb) {
						return errors.Errorf("mismatched number of inputs and segments in memBlock for segment %s", o.name)
					}

					o.inVals = vals[ind : ind+cap(o.inVals)]
					o.inputs = mb[i : i+len(o.inputs)]
					o.inputDims = dims[i : i+len(o.inputs)]
					o.numVpI = numValues[i : i+len(o.inputs)]
				}
			}

			if s == net.inSegments[0] {
				input = true
				inVInd = ind
				inInd = i
			} else if s == net.outSegments[0] {
				output = true
				outVInd = ind
				outInd = i
			}

			ind = end
		}
		if input {
			net.inputs = vals[inVInd : inVInd + len(net.inputs)]
			net.inSegments = mb[inInd : inInd + len(net.inSegments)]
		}
		if output {
			net.outputs = vals[outVInd : outVInd + len(net.outputs)]
			net.outSegments = mb[outInd : outInd + len(net.outSegments)]
		}
	}

	// as for the segments that aren't in memBlocks,
	// their values/inputs were already set by Add()
	// all that's left is to set inVals to be the right length
	var finishSetup func(*Segment)
	finishSetup = func(s *Segment) {
		if s.calc == inputsChanged {
			return
		}

		if len(s.inVals) < cap(s.inVals) {
			s.inVals = make([]float64, cap(s.inVals))
		}
		s.calc = inputsChanged

		for _, o := range s.outputs {
			finishSetup(o)
		}
	}

	for _, in := range net.inSegments {
		finishSetup(in)
	}

	return nil
}
