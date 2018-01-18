package smartlearn

import (
	"github.com/pkg/errors"
	"sync"
)

type Network struct {
	numSegments int

	inputs     []float64
	inSegments []*Segment

	outputs     []float64
	outSegments []*Segment
}

// no need for New() because new(Network) will be sufficient

// Add creates a new Segment and adds it to an existing network.
// dims is the dimensions of the Segment, for use by whichever SegmentType it is passed
// typ is the SegmentType for the Segment
// inputs
func (net *Network) Add(name string, typ SegmentType, dims []int, inputs ...*Segment) (*Segment, error) {
	if net.outSegments != nil {
		return nil, errors.Errorf("Can't add segment \"%s\" to network, network outputs have already been set (net.outSegments != nil)", name)
	} else if typ == nil {
		return nil, errors.Errorf("Can't add segment \"%s\" to network, typ (*SegmentType) == nil", name)
	}

	s := new(Segment)
	s.Name = "\"" + name + "\""
	s.id = net.numSegments
	net.numSegments++

	s.inputs = inputs
	s.Dims = dims
	s.typ = typ

	if len(inputs) <= 0 { // if it's an input segment
		// add it to the network inputs
		net.inSegments = append(net.inSegments, s)
		// s.InVals, InputDims, NumVpI = nil
	} else {
		totalNumInputs := 0
		s.NumVpI = make([]int, len(s.inputs))
		s.InputDims = make([][]int, len(s.inputs))
		for i, in := range inputs {
			in.outputs = append(in.outputs, s)

			s.InputDims[i] = in.Dims
			s.NumVpI[i] = len(in.Values)
			totalNumInputs += len(in.Values)
		}
		s.InVals = make([]float64, totalNumInputs)
	}

	if err := typ.SetValuesAndWeights(s); err != nil {
		return nil, errors.Wrapf(err, "Couldn't set values and weights for addition of segment %s\n", s.Name)
	} else if len(s.Values) <= 0 {
		return nil, errors.Errorf("Can't add segment %s, (*SegmentType).SetValuesAndWeights() created values with length <= 0 (%d)", s.Name, len(s.Values))
	}

	s.Deltas = make([]float64, len(s.Values))
	s.comLine = make(chan commandWrapper)
	result := make(chan error)
	go s.run(result)
	if err := <-result; err != nil {
		return nil, errors.Wrapf(err, "Couldn't run() segment %s as part of iitialization for addition to network", s.Name)
	}

	return s, nil
}

func (net *Network) SetOutputs(outputs ...*Segment) error {
	// check for error conditions and mark outputs with isOutput = true
	{
		if net.outSegments != nil {
			return errors.Errorf("Couldn't set network outputs - outputs have already been set")
		} else if len(outputs) <= 0 {
			return errors.Errorf("Couldn't set network outputs - len(outputs) == 0")
		} else if len(net.inSegments) <= 0 {
			return errors.Errorf("Couldn't set network outputs - network has no inputs")
		}

		net.outSegments = outputs

		numOutputs := 0
		for i, out := range net.outSegments {
			out.isOutput = true
			out.outValStart = numOutputs
			numOutputs += len(out.Values)
			if out.inputs == nil {
				return errors.Errorf("Couldn't set network outputs - segment %s (output %d) is also an input", out.Name, i)
			}
		}

		results := make([]chan error, len(net.inSegments))
		for i_, in_ := range net.inSegments {
			i, in := i_, in_
			results[i] = make(chan error)
			go func() { in.comLine <- commandWrapper{com: checkOutputs, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't set network outputs - checking for segments that don't affect outputs failed with input segment %s.\n", net.inSegments[i].Name)
			}
		}
	}

	// allocate the shared memory portion of the network
	{
		// initialize memBlocks with two elements so that it can hold the network inputs and outputs
		memBlocks := make([]*[]*Segment, 2)

		// set the first two memBlocks as network inputs : [0], and outputs : [1]
		// Note: This is technically not optimal, as there may be some cases where it would be necessary
		// to have the inputs and outputs in the same memBlock so that some intermediate semgent can
		// sequentially get input from both.
		{
			// copied from in/outSegments to avoid possible append bugs
			// shouldn't be a concern anyways, but this should be fine given that it only runs at startup
			zero, one := make([]*Segment, len(net.inSegments)), make([]*Segment, len(net.outSegments))
			memBlocks[0], memBlocks[1] = &zero, &one
			copy(*memBlocks[0], net.inSegments)
			copy(*memBlocks[1], net.outSegments)

			for ind := range memBlocks { // just inputs and outputs at this point
				mb := *(memBlocks[ind])
				for _, seg := range mb {
					seg.memBlock = memBlocks[ind]
				}
			}
		}

		// tell the rest of the segments to allocate, starting from the outputs
		results := make([]chan error, len(net.outSegments))
		for i_, out_ := range net.outSegments {
			i, out := i_, out_
			results[i] = make(chan error)
			go func() { out.comLine <- commandWrapper{allocate, results[i], []interface{}{&memBlocks}} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't allocate memory - error from network output segment %s\n", net.outSegments[i].Name)
			}
		}

		valueSets := make(map[*[]*Segment][]float64)
		var vsMux sync.Mutex

		// now that everything's allocated, tell all segments to finish allocating, starting from the outputs
		for i_, in_ := range net.outSegments {
			i, in := i_, in_
			results[i] = make(chan error)
			go func() { in.comLine <- commandWrapper{finishAllocating, results[i], []interface{}{valueSets, vsMux}} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't finish allocating memory - error from network input segment %s\n", net.inSegments[i].Name)
			}
		}

		// set network inputs and outputs to be identical to the memBlocks
		// uses memBlocks[:2] because the first two are guaranteed to contain the inputs, then the outputs
		for io, memBlock := range memBlocks[:2] {
			var netSl *[]*Segment
			var netVs *[]float64
			if io == 0 { // 0 is inputs
				netSl = &net.inSegments
				netVs = &net.inputs
			} else { // 1 is outputs
				netSl = &net.outSegments
				netVs = &net.outputs
			}

			numVs := 0
			for i := range *netSl {
				numVs += len((*netSl)[i].Values)
			}

			segBefore, vsBefore := -1, 0
			for i, seg := range *memBlock {
				if seg == (*netSl)[0] {
					segBefore = i
					break
				}

				vsBefore += len(seg.Values)
			}
			if segBefore == -1 {
				return errors.Errorf("Couldn't finalize allocating network inputs/outputs. Failed to find segment in its memBlock. io = %d", io)
			}
			*netSl = (*memBlock)[segBefore : segBefore+len(*netSl)]
			*netVs = valueSets[memBlock][vsBefore : vsBefore+numVs]
		}
	}

	// set all of the methods for each segment, now that the memory is all in the right place
	{
		results := make([]chan error, len(net.inSegments))
		for i_, in_ := range net.inSegments {
			i, in := i_, in_
			results[i] = make(chan error)
			go func() { in.comLine <- commandWrapper{com: setMethods, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't set network outputs - setting segment methods failed with input segment %s\n", net.inSegments[i].Name)
			}
		}
	}
	
	// send to all segments inputsChanged, indicating that the network has finished setting up
	{
		results := make([]chan error, len(net.inSegments))
		for i_, in_ := range net.inSegments {
			i, in := i_, in_
			results[i] = make(chan error)
			go func() { in.comLine <- commandWrapper{com: inputsChanged, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't set network outputs - sending 'inputsChanged' failed with input segment %s\n", net.inSegments[i].Name)
			}
		}
	}

	return nil
}
