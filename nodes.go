package badstudent

import "github.com/pkg/errors"

// returns the the Name of the Node, surrounded by double quotes
func (n *Node) String() string {
	if n.name != "" {
		return "\"" + n.name + "\""
	}

	return "<no name; type: " + n.typ.TypeString() + ">"
}

func (n *Node) Name() string {
	return n.name
}

// Returns whether or not the Node is an input node
func (n *Node) IsInput() bool {
	return num(n.inputs) == 0
}

// Returns whether or not the Node is an output node
func (n *Node) IsOutput() bool {
	return n.placeInOutputs >= 0
}

// SetDelay sets the amount of delay in the Node
//
// Constraints:
//
// The delay cannot be set after the network structure has been finalized
// by SetOutputs.
// Input Nodes cannot have delay because their values are set directly.
// Placeholder Nodes cannot have their delay set because their inputs are
// not yet known.
func (n *Node) SetDelay(delay int) error {
	if delay == n.Delay() {
		return nil
	}

	if n.host.stat >= finalized {
		return errors.Errorf("Network structure has already been finalized, cannot update Node delay")
	} else if n.IsPlaceholder() {
		return errors.Errorf("Node must no longer be a placeholder to set delay")
	} else if n.NumInputs() == 0 && delay > 0 {
		return errors.Errorf("Input nodes cannot have delay (delay = %d)", delay)
	}

	n.delay = make(chan []float64, delay)
	n.delayDeltas = make(chan []float64, delay)
	for i := 0; i < delay; i++ {
		n.delay <- make([]float64, len(n.values))
		n.delayDeltas <- make([]float64, len(n.values))
	}

	if delay != 0 {
		n.host.hasDelay = true
	} else {
		// update whether or not the network has delay
		n.host.hasDelay = false
		for _, t := range n.host.nodesByID {
			if t.HasDelay() {
				n.host.hasDelay = true
				break
			}
		}
	}

	return nil
}

func (n *Node) Delay() int {
	if n.delay == nil {
		return 0
	}

	return cap(n.delay)
}

func (n *Node) HasDelay() bool {
	return n.Delay() != 0
}

// Returns whether or not the Node is a placeholder Node
// Returns false if it never was, returns false if it has been replaced
func (n *Node) IsPlaceholder() bool {
	return n.host.stat == initialized && !n.completed
}

// returns the number of values that the node has
func (n *Node) Size() int {
	return len(n.values)
}

// Returns the value of the Node at the specified index
func (n *Node) Value(index int) float64 {
	return n.values[index]
}

// returns the value of the input to the Node at that index
//
// allows panics from index out of bounds and nil pointer
// a nil pointer means that the Node has no inputs
//
// binary searches to find the value
func (n *Node) InputValue(index int) float64 {
	return n.inputs.value(index)
}

// returns an unbuffered channel that goes through each value of the inputs
func (n *Node) InputIterator() chan float64 {
	return n.inputs.valueIterator()
}

func (n *Node) valueIterator() chan float64 {
	ch := make(chan float64)
	go func() {
		for _, v := range n.values {
			ch <- v
		}
		close(ch)
	}()

	return ch
}

// returns the 'delta' of the value at of the node at the given index
//
// this is a SIMPLE FUNCTION. does not check if deltas have
// been calculated before running.
func (n *Node) Delta(index int) float64 {
	return n.deltas[index]
}

// returns the number of nodes that the node receives input from
func (n *Node) NumInputNodes() int {
	return num(n.inputs)
}

// returns the total number of input values to the node
func (n *Node) NumInputs() int {
	return n.inputs.size()
}

// returns the size of the given input to the node
func (n *Node) InputSize(index int) int {
	return n.inputs.nodes[index].Size()
}

// returns a single slice containing a copy of all of
// the input values to the node, in order
func (n *Node) CopyOfInputs() []float64 {
	return n.inputs.getValues(true)
}
