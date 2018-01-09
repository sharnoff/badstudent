package smartlearn

import (
	"github.com/pkg/errors"
	"sync"
)

type Segment struct {
	Name string

	Values    []float64
	Weights   []float64
	Deltas    []float64
	InVals    []float64 // set of all input values to this segment
	InputDims [][]int
	NumVpI    []int // number of values from each input to this segment
	Dims      []int

	id int // for future use in saving to a file / getting from a file

	inputs      []*Segment
	outputs     []*Segment
	isOutput    bool
	outValStart int // the index in net.outputs that the values of the segment start at

	inputsIdentical bool       // true if InVals and inputs point to the same memory space as inputs[].Values
	memBlock        *[]*Segment // the set of other segments (and itself) that this is located with
	valueSet        []float64

	prog    map[command]progress
	deltasProg []progress
	progMux sync.Mutex
	comLine chan commandWrapper

	// these methods are what are set by 'setMethods' and run more than just the SemgentType
	// portion of the methods
	evaluate    func(chan resultWrapper, []interface{})
	inputDeltas func(chan resultWrapper, []interface{})
	adjust      func(chan resultWrapper, []interface{})
	typ         SegmentType
}

// 'origin' is the command that was executed to return this
// 'res' is short for 'result'
// 'crit' is whether or not the error error (if there was one) was critical
// 		if crit is true, the segment will mark that process as 'done'
type resultWrapper struct {
	origin command
	res    error
	crit   bool
}

type progress int

const (
	notStarted progress = iota
	inProgress progress = iota
	done       progress = iota
)

// need to implement some way to handle errors based on whether they're critical or not
func (s *Segment) run(init chan error) {
	if s.comLine == nil {
		init <- errors.Errorf("Can't run segment, comLine hasn't been initialized")
		close(init)
		return
	} else {
		init <- nil
		close(init)
	}

	s.progMux.Lock()
	s.prog = make(map[command]progress)
	s.progMux.Unlock()
	// progMux doesn't apply to deltasProg
	s.deltasProg = make(map[*Segment]progress)

	resLine := make(chan resultWrapper)
	funcs := map[command]func(*Segment, chan resultWrapper, []interface{}){
		checkOutputs:     func(s *Segment, res chan resultWrapper, aux []interface{}) { s.checkOutputs(res, aux) },
		allocate:         func(s *Segment, res chan resultWrapper, aux []interface{}) { s.allocate(res, aux) },
		finishAllocating: func(s *Segment, res chan resultWrapper, aux []interface{}) { s.finishAllocating(res, aux) },
		setMethods:       func(s *Segment, res chan resultWrapper, aux []interface{}) { s.setMethods(res, aux) },
		inputsChanged:    func(s *Segment, res chan resultWrapper, aux []interface{}) { s.inputsChanged(res, aux) },
		evaluate:         func(s *Segment, res chan resultWrapper, aux []interface{}) { s.evaluate(res, aux) },
		deltas:           func(s *Segment, res chan resultWrapper, aux []interface{}) { s.deltas(res, aux) },
		inputDeltas:      func(s *Segment, res chan resultWrapper, aux []interface{}) { s.inputDeltas(res, aux) },
		adjust:           func(s *Segment, res chan resultWrapper, aux []interface{}) { s.adjust(res, aux) },
	}

	// this doesn't need to be initialized because we can append to a nil slice
	returnChs := make(map[command][]chan error)

	// this is used for the one exception to this system:
	// inputDeltas needs to be individually done each time 
	// deltasWaiting follows LiFo
	deltasWaiting := make([]commandWrapper, 0, len(s.inputs))

	pastSetup := false

	select {
	case c := <-s.comLine:
		if c.com == 0 { // if nil
			go func() { c.res <- errors.Errorf("Couldn't run command to segment %s, com = nil.", s.Name) }()
			break
		} else if !c.com.isSetup() && !c.com.isRunning() {
			go func() {
				c.res <- errors.Errorf("Coudln't run command to segment %s, unknown command: %s", s.Name, c.com.String())
			}()
			break
		} else if pastSetup != c.com.isRunning() {
			go func() {
				c.res <- errors.Errorf("Couldn't run command to segment %s, command doesn't match phase (setup/running) running: %t. (command: %s)", s.Name, pastSetup, c.com.String())
			}()
			break
		}

		// this is the one exception
		if c.com == inputDeltas {
			s.progMux.Lock()
			if s.prog[inputDeltas] != notStarted {
				s.progMux.Unlock()
				deltasWaiting = append(deltasWaiting, c)
			} else {
				s.prog[inputDeltas] = inProgress
				s.progMux.Unlock()
				go funcs[c.com](s, resLine, c.aux)
			}

			break
		}

		s.progMux.Lock()
		if s.prog[c.com] == done {
			s.progMux.Unlock()
			go func() { c.res <- nil }()
			break
		}
		returnChs[c.com] = append(returnChs[c.com], c.res)
		if s.prog[c.com] == inProgress {
			s.progMux.Unlock()
			break
		}
		s.prog[c.com] = inProgress
		s.progMux.Unlock()

		go funcs[c.com](s, resLine, c.aux)
	case r := <-resLine:
		var err error
		if r.res != nil {
			if r.crit {
				err = errors.Wrapf(r.res, "Critical error from s.%s() for segment %s\n", r.origin.String(), s.Name)
			} else {
				err = errors.Wrapf(r.res, "Non-critical error from s.%s() for segment %s\n", r.origin.String(), s.Name)
			}
		}
 
		if r.origin == inputDeltas {
			s.progMux.Lock()
			s.prog[inputDeltas] = notStarted
			s.progMux.Unlock()

			go func() { returnChs[r.origin[0]] <- err }

			if len(deltasWaiting) != 0 {
				go func(){ s.comLine <- deltasWaiting[len(deltasWaiting) - 1] }()
				deltasWaiting = deltasWaiting[:len(deltasWaiting) - 1] // slice one element off the end
			}
		} else {
			s.progMux.Lock()
			if r.res != nil && r.crit {
				s.prog[r.origin] = notStarted
			} else {
				s.prog[r.origin] = done
			}
			s.progMux.Unlock()

			for _, ch := range returnChs[r.origin] {
				go func() { ch <- err }()
			}

			if r.origin.isSetup() && r.origin.isRunning() {
				pastSetup = true
			}
		}
	}
}

// checkOutputs does nothing with aux
// recurses towards outputs
// checkOutputs just makes sure that there are no segments that have no outputs but aren't outputs
func (s *Segment) checkOutputs(result chan resultWrapper, aux []interface{}) {
	// perform the actual checking of the outputs
	if !s.isOutput && s.outputs == nil {
		err := errors.Errorf("Segment %s has no outputs but is not an output (s.outputs == nil && !s.isOutput)", s.Name)
		result <- resultWrapper{res: err, origin: checkOutputs, crit: true}
		return
	}

	// recurse towards outputs
	results := make([]chan error, len(s.outputs))
	for i, out := range s.outputs {
		results[i] = make(chan error)
		go func() { out.comLine <- commandWrapper{com: checkOutputs, res: results[i]} }()
	}
	for i, ch := range results {
		if err := <-ch; err != nil {
			err = errors.Wrapf(err, "Error in attempting to check outputs of segment %s from segment %s\n", s.outputs[i].Name, s.Name)
			result <- resultWrapper{origin: checkOutputs, res: err}
		}
	}

	result <- resultWrapper{origin: checkOutputs}
	return
}

// allocate 'returns' error if aux[0] is not type *[]*[]*Segment
// allocate calls checkOutputs (through s.comLine) if s.prog[checkOutpus] != done
// recurses towards inputs
func (s *Segment) allocate(result chan resultWrapper, aux []interface{}) {
	// if it hasn't checked outputs, do that
	s.progMux.Lock()
	if s.prog[checkOutputs] != done {
		s.progMux.Unlock()
		res := make(chan error)
		s.comLine <- commandWrapper{com: checkOutputs, res: res}
		if err := <-res; err != nil {
			err = errors.Wrapf(err, "Couldn't check outputs before trying to allocate segment %s\n", s.Name)
			result <- resultWrapper{origin: allocate, res: err}
			return
		}
	} else {
		s.progMux.Unlock()
	}

	// set memBlocks to aux[0], so long as it is type *[]*[]*Segment
	var memBlocks *[]*[]*Segment
	var ok bool

	// if it's already been allocated (which is likely if it's either an input or an output segment)
	// then recurse downwards
	// we have to do this here so that we don't skip variable delcarations
	if s.memBlock != nil {
		goto Recurse
	}

	if len(aux) == 0 {
		err := errors.Errorf("Can't allocate segment %s, len(aux) == 0. Allocate requires aux[0] to be type %T", s.Name, memBlocks)
		result <- resultWrapper{origin: allocate, res: err}
		return
	} else if memBlocks, ok = aux[0].(*[]*[]*Segment); !ok {
		err := errors.Errorf("Can't allocate segment %s, aux[0] is not type %T (%T)", s.Name, memBlocks, aux[0])
		result <- resultWrapper{origin: allocate, res: err}
	}

	// the bulk of the allocating:
	{
		// check to see how many of the inputs have already been allocated
		indAlloc := make([]int, 0, len(s.inputs))
		for i, in := range s.inputs {
			if in.memBlock != nil {
				indAlloc = append(indAlloc, i)
			}
		}

		// if nothing's been allocated, then allocate the whole set and be done
		if len(indAlloc) == 0 {
			// add the whole set of inputs to memBlock
			newMb := make([]*Segment, len(s.inputs))
			// probably not necessary, but good practice considering this only gets run once
			copy(newMb, s.inputs)

			for _, in := range s.inputs {
				in.memBlock = &newMb
			}

			*memBlocks = append(*memBlocks, &newMb)
			s.inputsIdentical = true
		} else {
			// in general for this section:
			// if the inputs can be added to the memBlock of the ones already allocated,
			// do that
			// 'goto Recurse' is used instead of returning a non-error,
			// because the inputs still need to allocate()

			// check that all of the inputs that have already been allocated are in the same segment
			mb := s.inputs[indAlloc[0]].memBlock
			for _, i := range indAlloc {
				if s.inputs[i].memBlock != mb { // compare equality
					goto Recurse
				}
			}

			// now that we know that all of the inputs that have already been allocated are in the same
			// memBlock, check that they are aligned (continuous) in physical memory

			// check for continuity in indAlloc
			// this is to make sure that the rest of the inputs could actually be added
			for i := 0; i < len(indAlloc)-1; i++ {
				if indAlloc[i+1] != indAlloc[i]+1 {
					goto Recurse
				}
			}

			// find the index in mb of the first allocated input (indAlloc[0])
			start := -1
			for i, seg := range *mb {
				if seg == s.inputs[indAlloc[0]] {
					start = i
					break
				}
			}
			if start == -1 {
				err := errors.Errorf("Couldn't allocate semgent %s due to critical error - first allocated input segment (%s) not in own memBlock", s.Name, s.inputs[indAlloc[0]].Name)
				result <- resultWrapper{origin: allocate, res: err, crit: true}
				return
			}

			// check that set of inputs in indAlloc are physically
			// aligned in the same way in mb
			for i := 0; i < len(indAlloc); i++ {
				if (*mb)[start+i] != s.inputs[indAlloc[i]] {
					goto Recurse
				}
			}

			// check that it can be appended to either the beginning or the end of mb (or both)
			if indAlloc[0] != 0 && start != 0 {
				// if it can't append below the indAlloc set
				goto Recurse
			} else if indAlloc[len(indAlloc)-1] != len(s.inputs)-1 && start+len(indAlloc) != len(*mb) {
				// if it can't append above the indAlloc set
				goto Recurse
			}

			less := make([]*Segment, indAlloc[0])
			copy(less, s.inputs[:indAlloc[0]])
			*mb = append(less, (*mb)...)
			*mb = append(*mb, s.inputs[indAlloc[len(indAlloc)-1]:]...)
			for _, in := range s.inputs {
				in.memBlock = mb
			}

			s.inputsIdentical = true
		}
	}
Recurse:
	// recurse towards inputs
	// this CANNOT BE MULTI-THREADED because it would cause segments to be put into multiple memBlocks
	for _, in := range s.inputs {
		res := make(chan error)
		in.comLine <- commandWrapper{com: allocate, res: res, aux: []interface{}{memBlocks}}
		if err := <-res; err != nil {
			err = errors.Wrapf(err, "Failed to allocate input segment %s to segment %s\n", s.Name, in.Name)
		}
	}
	result <- resultWrapper{origin: allocate}
	return
}

// finishAllocating 'returns' error if aux[0] is not type map[*[]*Segment][]float64
// or if aux[1] is not type sync.Mutex
// if allocate() hasn't been run yet, finishAllocating passes aux[2:] to allocate()
// recurses before running towards inputs
func (s *Segment) finishAllocating(result chan resultWrapper, aux []interface{}) {
	if len(aux) == 0 {
		err := errors.Errorf("Can't finish allocating segment %s, len(aux) == 0. See documentation for correct values of aux for finishAllocating()", s.Name)
		result <- resultWrapper{origin: finishAllocating, res: err}
		return
	}

	s.progMux.Lock()
	if s.prog[allocate] != done {
		s.progMux.Unlock()
		res := make(chan error)
		s.comLine <- commandWrapper{com: allocate, res: res, aux: aux[2:]}
		if err := <-res; err != nil {
			err = errors.Wrapf(err, "Couldn't allocate segment %s, attempt to allocate segment failed. 'aux' sent to allocate() is equal to aux[2:]\n", s.Name)
			result <- resultWrapper{origin: finishAllocating, res: err}
			return
		}
	} else {
		s.progMux.Unlock()
	}

	// make sure that:
	//   aux[0] is type map[*[]*Segment][]float64
	//   aux[1] is type sync.Mutex
	var valueSets map[*[]*Segment][]float64
	var vsMux sync.Mutex
	var ok bool
	if valueSets, ok = aux[0].(map[*[]*Segment][]float64); !ok {
		err := errors.Errorf("Can't finish allocating segment %s, aux[0] is not type %T (%T)", valueSets, aux[0])
		result <- resultWrapper{origin: finishAllocating, res: err}
		return
	} else if vsMux, ok = aux[1].(sync.Mutex); !ok {
		err := errors.Errorf("Can't finish allocating segment %s, aux[1] is not type %T (%T)", vsMux, aux[1])
		result <- resultWrapper{origin: finishAllocating, res: err}
		return
	}

	// recurse before starting, so that all inputs' valueSets are already set
	// recurses towards inputs
	results := make([]chan error, len(s.inputs))
	for i, in := range s.inputs {
		results[i] = make(chan error)
		go func() { in.comLine <- commandWrapper{com: finishAllocating, res: results[i], aux: aux} }()
	}
	for i, ch := range results {
		if err := <-ch; err != nil {
			err = errors.Wrapf(err, "Couldn't finish allocating segment %s - allocating input segment %s failed.\n", s.Name, s.inputs[i].Name)
			result <- resultWrapper{origin: finishAllocating, res: err}
			return
		}
	}

	// finish allocating - set up the segment's values, inValues, and inputs with their new spaces in memory
	vsMux.Lock()
	if valueSets[s.memBlock] == nil {
		var sum int
		for _, seg := range *(s.memBlock) {
			sum += len(seg.Values)
		}
		valueSets[s.memBlock] = make([]float64, sum)
	}
	vsMux.Unlock()

	// find how many other segments and values there are before this segment / its values
	segBefore, vsBefore := -1, 0
	for i, seg := range (*s.memBlock) { // it's okay to run this multiple times because this is during setup
		if seg == s {
			segBefore = i
			break
		}

		vsBefore += len(seg.Values)
	}
	if segBefore == -1 {
		err := errors.Errorf("Couldn't finish allocating segment %s, failed to find segment in own memBlock", s.Name)
		result <- resultWrapper{origin: finishAllocating, res: err, crit: true}
		return
	}

	s.Values = valueSets[s.memBlock][vsBefore : vsBefore+len(s.Values)]

	// set inputs
	if s.inputsIdentical {
		segBefore, vsBefore = -1, 0
		for i, seg := range *(s.inputs[0].memBlock) {
			if seg == s.inputs[0] {
				segBefore = i
				break
			}

			vsBefore += len(seg.Values)
		}
		if segBefore == -1 {
			err := errors.Errorf("Couldn't finish allocating segment %s, failed to find 0th input (%s) in input's own memBlock", s.Name, s.inputs[0].Name)
			result <- resultWrapper{origin: finishAllocating, res: err, crit: true}
			return
		}

		s.InVals = valueSets[s.inputs[0].memBlock][vsBefore : vsBefore+len(s.InVals)]
		s.inputs = (*s.inputs[0].memBlock)[segBefore : segBefore + len(s.inputs)]
	}

	result <- resultWrapper{origin: finishAllocating}
	return
}

// setMethods does nothing with aux, except for passing it to finishAllocating() if
// it hasn't run yet.
// recurses towards outputs
func (s *Segment) setMethods(result chan resultWrapper, aux []interface{}) {
	// if it hasn't finished allocating, pass aux to finishAllocating()
	s.progMux.Lock()
	if s.prog[finishAllocating] != done {
		s.progMux.Unlock()
		res := make(chan error)
		s.comLine <- commandWrapper{com: finishAllocating, res: res, aux: aux}
		if err := <-res; err != nil {
			err = errors.Wrapf(err, "Couldn't allocate segment %s, attempt to allocate segment failed. 'aux' sent to allocate() is equal to aux[2:]\n", s.Name)
			result <- resultWrapper{origin: finishAllocating, res: err}
			return
		}
	} else {
		s.progMux.Unlock()
	}

	// set the methods (which are actually members of the Segment struct)
	evaluateFunc, err := s.typ.EvaluateFunc(s)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't set methods for segment %s, EvaluateFunc() on s.typ (type %T) failed\n", s.Name, s.typ)
		result <- resultWrapper{origin: setMethods, res: err}
		return
	}
	inputDeltasFunc, err := s.typ.InputDeltasFunc(s)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't set methods for segment %s, InputDeltasFunc() on s.typ (type %T) failed\n", s.Name, s.typ)
		result <- resultWrapper{origin: setMethods, res: err}
		return
	}
	adjustFunc, err := s.typ.AdjustFunc(s)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't set methods for segment %s, AdjustFunc() on s.typ (type %T) failed\n", s.Name, s.typ)
		result <- resultWrapper{origin: setMethods, res: err}
		return
	}

	s.evaluate = func(result chan resultWrapper, aux []interface{}) {
		s.progMux.Lock()
		if s.prog[inputsChanged] == notStarted {
			s.progMux.Unlock()
			result <- resultWrapper{origin: evaluate}
			return
		} else if s.prog[inputsChanged] == inProgress { // make sure that prog[inputsChanged] == done
			s.progMux.Unlock()
			res := make(chan error)
			s.comLine <- commandWrapper{com: inputsChanged, res: res}
			if err := <-res; err != nil {
				err = errors.Wrapf(err, "Couldn't evaluate segment %s - failed to finish notifying that inputsChanged\n", s.Name)
				result <- resultWrapper{origin: evaluate, res: err}
				return
			}
		} else {
			s.progMux.Unlock()
		}

		// recurse downwards, so that inVals have been calculated
		results := make([]chan error, len(s.inputs))
		for i, in := range s.inputs {
			results[i] = make(chan error)
			go func() { in.comLine <- commandWrapper{com: evaluate, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				err = errors.Wrapf(err, "Couldn't evaluate segment %s, evaluating input segment %s failed.\n", s.Name, s.inputs[i].Name)
				result <- resultWrapper{origin: evaluate, res: err}
				return
			}
		}

		if !s.inputsIdentical {
			i := 0
			for _, in := range s.inputs {
				copy(s.InVals[i:], in.Values) // this is fine because copy just uses
				i += len(in.Values)
			}
		}

		err := evaluateFunc()
		if err != nil {
			result <- resultWrapper{origin: evaluate, res: err, crit: true}
			return
		}

		s.progMux.Lock()
		s.prog[inputsChanged] = notStarted
		s.progMux.Unlock()

		result <- resultWrapper{origin: evaluate}
		return
	}

	s.inputDeltas = func(result chan resultWrapper, aux []interface{}) {
		if len(aux) < 2 {
			err := errors.Errorf("Couldn't get input deltas of segment %s, len(aux) < 2 (inputDeltas() requires aux[0] to be type *Segment) and aux[1] to be type []float64", s.Name)
			result <- resultWrapper{origin: inputDeltas, res: err}
			return
		}

		s.progMux.Lock()
		if s.prog[deltas] != done {
			s.progMux.Unlock()
			res := make(chan error)
			s.comLine <- commandWrapper{com: deltas, res: res, aux: aux[2:]}
			if err := <-res; err != nil {
				err = errors.Wrapf(err, "Couldn't get input deltas of segment %s, getting own deltas failed (Note: inputDeltas passes aux[2:] to deltas\n", s.Name)
				result <- resultWrapper{origin: inputDeltas, res: err}
				return
			}
		} else {
			s.progMux.Unlock()
		}

		var selectInput *Segment
		var inputDs []float64

		var ok bool
		if selectInput, ok = aux[0].(*Segment); !ok {
			err := errors.Errorf("Couldn't get input deltas of segment %s, aux[0] should be type %T (was type %T)", s.Name, selectInput, aux[0])
			result <- resultWrapper{origin: inputDeltas, res: err}
			return
		} else if inputDs, ok = aux[1].([]float64); !ok {
			err := errors.Errorf("Couldn't get input deltas of segment %s, aux[1] should be type %T (was type %T)", s.Name, selectInput, aux[1])
			result <- resultWrapper{origin: inputDeltas, res: err}
			return
		}

		// find the index of the selected input
		inputIndex := -1
		for i := range s.inputs {
			if s.inputs[i] == selectInput {
				inputIndex = i
				break
			}
		}
		if inputIndex == -1 {
			err := errors.Errorf("Couldn't get input deltas of segment %s, given segment %s to get deltas for is not an input of %s", s.Name, selectInput.Name, s.Name)
			result <- resultWrapper{origin: inputDeltas, res: err}
			return
		}

		// can't equal inProgress because there can't be two of these going at the same time
		// if this one has already been done, return nil.
		if s.deltasProg[inputIndex] == done {
			result <- resultWrapper{origin: inputDeltas}
			return
		}

		err := inputDeltasFunc(inputIndex, inputDs)
		if err != nil {
			result <- resultWrapper{origin: inputDeltas, res: err}
			return
		}

		s.deltasProg[inputIndex] = done
		result <- resultWrapper{origin: inputDeltas}
		return
	}

	s.adjust = func(result chan resultWrapper, aux []interface{}) {
		if len(aux) == 0 {
			err := errors.Errorf("Can't adjust segment %s, len(aux) == 0", s.Name)
			result <- resultWrapper{origin: adjust, res: err}
			return
		}

		s.progMux.Lock()
		if s.prog[deltas] != done {
			s.progMux.Unlock()
			res := make(chan error)
			s.comLine <- commandWrapper{com: deltas, res: res, aux: aux[2:]}
			if err := <-res; err != nil {
				err = errors.Wrapf(err, "Couldn't adjust segment %s, getting own deltas failed (Note: adjust passes aux[1:] to deltas\n", s.Name)
				result <- resultWrapper{origin: adjust, res: err}
				return
			}
		} else {
			s.progMux.Unlock()
		}

		var learningRate float64
		var ok bool
		if learningRate, ok = aux[0].(float64); !ok {
			err := errors.Errorf("Can't adjust segment %s, aux[0] is not type %T (type %T)", s.Name, learningRate, aux[0])
			result <- resultWrapper{origin: adjust, res: err}
			return
		}

		err := adjustFunc(learningRate)
		if err != nil {
			result <- resultWrapper{origin: adjust, res: err}
			return
		}

		// recurse towards inputs
		results := make([]chan error, len(s.inputs))
		for i, in := range s.inputs {
			results[i] = make(chan error)
			go func(){ in.comLine <- commandWrapper{adjust, results[i], aux} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				err = errors.Wrapf(err, "Couldn't adjust segment %s, recursing to input %s failed\n", s.Name, s.inputs[i].Name)
				result <- resultWrapper{origin: adjust, res: err}
				return
			}
		}

		result <- resultWrapper{origin: adjust}
		return
	}

	// recurse towards outputs
	results := make([]chan error, len(s.outputs))
	for i, out := range s.outputs {
		results[i] = make(chan error)
		go func() { out.comLine <- commandWrapper{com: setMethods, res: results[i], aux: aux} }()
	}
	for i, ch := range results {
		if err := <-ch; err != nil {
			err = errors.Wrapf(err, "Failed after setting methods of %s - setting output segment %s failed.\n", s.Name, s.outputs[i].Name)
			result <- resultWrapper{origin: setMethods, res: err}
			return
		}
	}

	result <- resultWrapper{origin: setMethods}
	return
}

// this function is not actually safe, because there could be other things going on,
// but a fix would be time-intensive, and it's realistically not a problem.
// inputsChanged does nothing with aux
// sets prog[evaluate, deltas, inputDeltas, adjust] to notStarted
// recurses towards outputs
func (s *Segment) inputsChanged(result chan resultWrapper, aux []interface{}) {
	s.progMux.Lock()
	s.prog[evaluate] = notStarted
	s.prog[deltas] = notStarted
	s.deltasProg = make([]progress, len(s.inputs))
	s.prog[adjust] = notStarted
	s.progMux.Unlock()

	results := make([]chan error, len(s.outputs))
	for i, out := range s.outputs {
		results[i] = make(chan error)
		go func() { out.comLine <- commandWrapper{com: inputsChanged, res: results[i]} }()
	}
	for i, ch := range results {
		if err := <-ch; err != nil {
			err = errors.Wrapf(err, "Failed to notify of inputs changed in segment %s while recursing to output segment %s\n", s.Name, s.outputs[i].Name)
			result <- resultWrapper{origin: inputsChanged, res: err}
			return
		}
	}
}

// evaluate does nothing with aux
// recurses before running towards inputs
// func (s *Segment) evaluate(result chan resultWrapper, aux []interface{})

// deltas asserts that aux[0] is type []float64
// passes certain values inputDeltas through aux, with aux put at the end
// calls inputDeltas on outputs before running
func (s *Segment) deltas(result chan resultWrapper, aux []interface{}) {
	s.progMux.Lock()
	if s.prog[evaluate] != done {
		s.progMux.Unlock()
		res := make(chan error)
		s.comLine <- commandWrapper{com: evaluate, res: res}
		if err := <-res; err != nil {
			err = errors.Wrapf(err, "Coudln't get deltas of segment %s, evaluating failed\n", s.Name)
			result <- resultWrapper{origin: deltas, res: err}
			return
		}
	} else {
		s.progMux.Unlock()
	}

	var targetOutputs []float64
	var ok bool
	if len(aux) == 0 {
		err := errors.Errorf("Can't get deltas for segment %s, len(aux) == 0", s.Name)
		result <- resultWrapper{origin: deltas, res: err}
		return
	} else if targetOutputs, ok = aux[0].([]float64); !ok {
		err := errors.Errorf("Couldn't get deltas of segment %s, aux[0] should be type %T (was type %T)", s.Name, targetOutputs, aux[0])
		result <- resultWrapper{origin: deltas, res: err}
		return
	}

	results := make([]chan error, len(s.outputs))
	var ds [][]float64
	if s.isOutput {
		ds = make([][]float64, len(s.outputs)+1)
	} else {
		ds = make([][]float64, len(s.outputs))
	}

	for i, out := range s.outputs {
		results[i] = make(chan error)
		ds[i] = make([]float64, len(s.Values))
		go func(){out.comLine <- commandWrapper{inputDeltas, results[i], []interface{}{i, ds[i], targetOutputs}}}()
	}
	for i, ch := range results {
		if err := <-ch; err != nil {
			err = errors.Wrapf(err, "Couldn't get deltas of segment %s, getting input deltas of output segment %s failed.\n", s.Name, s.outputs[i].Name)
			result <- resultWrapper{origin: deltas, res: err}
			return
		}
	}

	if s.isOutput {
		if s.outValStart+len(s.Values) > len(targetOutputs) {
			err := errors.Errorf("Couldn't get deltas for segment %s, either segment outValStart was misinformed or provided target outputs have inadequate length", s.Name)
			result <- resultWrapper{origin: deltas, res: err}
			return
		}

		ds[len(s.outputs)] = make([]float64, len(s.Values))
		for i := range s.Values {
			// assumes it is optimzing for the squared error function
			ds[len(s.outputs)][i] = s.Values[i] - targetOutputs[s.outValStart+i]
		}
	}

	// sum all of the slices in ds and set deltas to that
	for i := range s.Values {
		var sum float64
		for _, d := range ds {
			sum += d[i]
		}
		s.Deltas[i] = sum
	}

	result <- resultWrapper{origin: deltas}
	return
}

// inputDeltas asserts that aux[0] is type *Segment and aux[1] is type []float64
// aux[0] should be the target segment whose deltas (from this segment)
// aux[1] should be the slice of float that deltas are being written to (to eventually be summed)
// calls deltas() on itself before running
// func (s *Segment) inputDeltas(result chan resultWrapper, aux []interface{})

// adjust asserts that aux[0] is type float64
// if deltas hasn't run yet, it passes aux[1:] to deltas
// recurses towards inputs
// func (s *Segment) adjust(result chan resultWrapper, aux []interface{})
