package badstudent

// String returns the the Name of the Node, surrounded by double quotes
func (n *Node) String() string {
	if n.name != "" {
		return "\"" + n.name + "\""
	}

	return "<unnamed; type: " + n.typ.TypeString() + ">"
}

func (n *Node) Name() string {
	return n.name
}

func (n *Node) IsInput() bool {
	return num(n.inputs) == 0
}

func (n *Node) IsOutput() bool {
	return n.placeInOutputs >= 0
}

// Delay returns the amount of time-step delay between a hypothetical change in
// inputs and a change in outputs for a Node. If the Node does not have delay, this
// will return 0.
func (n *Node) Delay() int {
	if n.delay == nil {
		return 0
	}

	return cap(n.delay)
}

// HasDelay returns whether or not the Node's outputs are delayed from its inputs.
func (n *Node) HasDelay() bool {
	return n.Delay() != 0
}

// IsPlaceholder returns whether or not the Node is a placeholder Node, i.e. it must
// be replaced before Network finalization. This only returns true if it is
// currently a placeholder.
func (n *Node) IsPlaceholder() bool {
	return n.host.stat == initialized && !n.completed
}

// Size returns the number of values that the node has
func (n *Node) Size() int {
	return len(n.values)
}

// HP returns the value of the given HyperParameter at the current iteration. If an
// unknown HyperParameter is asked for, HP will panic with a nil pointer
// dereference. This can be avoided by correct usage of Optimizer Needs.
func (n *Node) HP(name string) float64 {
	return n.hyperParams[name].Value(n.host.iter)
}

// Value returns the value of the Node at the specified index. Value will panic if
// given an index out of bounds.
func (n *Node) Value(index int) float64 {
	return n.values[index]
}

// InputValue returns the value of the input to the Node at that index
//
// NB: This function binary searches among the input Nodes to get the value. If
// there are many input Nodes, this will quickly become a function that is no longer
// efficient to call en masse.
//
// InputValue allows panics from index out of bounds and nil pointers.
func (n *Node) InputValue(index int) float64 {
	return n.inputs.value(index)
}

// InputIterator returns an unbuffered channel that goes through each value of the
// inputs
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

// Delta returns the 'delta' of the value at of the node at the given index (the
// derivative w.r.t. the total cost).
//
// This will not check if deltas have actually been calculated. Except in special
// cases, they will have been.
func (n *Node) Delta(index int) float64 {
	return n.deltas[index]
}

// NumInputNodes returns the number of Nodes from which the Node recieves input.
func (n *Node) NumInputNodes() int {
	return num(n.inputs)
}

// NumInputs returns the total number of input values to the node
func (n *Node) NumInputs() int {
	return n.inputs.size()
}

// InputSize returns the size of the given input to the node
func (n *Node) InputSize(index int) int {
	return n.inputs.nodes[index].Size()
}

// CopyOfINputs returns a single slice containing a copy of all of the input values
// to the node, in order
func (n *Node) CopyOfInputs() []float64 {
	return n.inputs.getValues(true)
}
