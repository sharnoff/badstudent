package badstudent

import ()

// returns the the Name of the Node, surrounded by double quotes
func (n *Node) String() string {
	return "\"" + n.Name + "\""
}

// Returns whether or not the Node is an input node
func (n *Node) IsInput() bool {
	return num(n.inputs) == 0
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
