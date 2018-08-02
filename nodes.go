package badstudent

import (
	"sort"
	"sync"
)

// returns the the Name of the Node, surrounded by double quotes
func (n *Node) String() string {
	return "\"" + n.Name + "\""
}

// returns the number of values that the node has
func (n *Node) Size() int {
	return len(n.values)
}

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
	greaterThan := func(i int) bool {
		return index < n.numInputs[i]
	}

	i := sort.Search(len(n.inputs), greaterThan)

	if i > 0 {
		index -= n.numInputs[i-1]
	}

	return n.inputs[i].values[index]
}

// returns an unbuffered channel that goes through each value of the inputs
func (n *Node) InputIterator() chan float64 {
	ch := make(chan float64)
	go func() {
		for _, in := range n.inputs {
			for _, v := range in.values {
				ch <- v
			}
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
	return len(n.inputs)
}

// returns the total number of input values to the node
func (n *Node) NumInputs() int {
	if len(n.inputs) == 0 {
		return 0
	}

	return n.numInputs[len(n.inputs) - 1]
}

// returns the size of the given input to the node
func (n *Node) InputSize(index int) int {
	return n.inputs[index].Size()
}

// Returns the number of values that provide input to the node
// before the node at the given index does
// Can provide index of *Node.NumInputNodes(), which will return total number of inputs
//
// will allow the index out of bounds panic to go through if index is not in range
func (n *Node) PreviousInputs(index int) int {
	if index == 0 {
		return 0
	} else {
		return n.numInputs[index - 1]
	}
}

// returns a single slice containing a copy of all of
// the input values to the node, in order
func (n *Node) CopyOfInputs() []float64 {

	c := make([]float64, n.NumInputs())
	start := 0
	for _, in := range n.inputs {
		copy(c[start : start + in.Size()], in.values)
		start += in.Size()
	}

	return c
}
