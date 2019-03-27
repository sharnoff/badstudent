package badstudent

import (
	"github.com/sharnoff/tensors"
	"fmt"
)

// String offers a universal method of gaining information about a Node without printing all of its
// fields. String returns the Node's Name, unless it is an empty string, in which case it returns:
//	<id: %d; type: %s>
// where %d is the Node's id, and %s is the name of its Operator.
// If the Node has no Operator (because it's an input Node), it'll return:
//	<Is Input, id: %d>
// Finally, if given a Node that is nil, String will return:
//	<nil>
func (n *Node) String() string {
	if n == nil {
		return "<nil>"
	}

	if n.name != "" {
		return "\"" + n.name + "\""
	} else if n.isPlaceholder() {
		return fmt.Sprintf("<Is Placeholder, id: %d>", n.id)
	} else if n.op == nil {
		return fmt.Sprintf("<Is Input, id: %d>", n.id)
	}

	return fmt.Sprintf("<id: %d, Operator: %s>", n.id, n.op.TypeString())
}

// Name returns the name of the given Node. This may sometimes be an empty string. The name of a
// Node can be set with *Node.SetName()
func (n *Node) Name() string {
	return n.name
}

// ID returns the non-negative integer given to the Node as a member of its Network. IDs are unique
// within Networks.
func (n *Node) ID() int {
	return n.id
}

// IsInput returns whether or not the Node is an input Node. Input Nodes will not have Operators.
func (n *Node) IsInput() bool {
	// Because placeholders mark themselves by their inputs being non-nil, only input Nodes have
	// nil inputs.
	return n.inputs == nil
}

// IsOutput returns whether or not the Node is an output Node.
func (n *Node) IsOutput() bool {
	return n.outputIndex >= 0
}

// Delay returns the amount of time-step delay between a hypothetical change in inputs and a change
// in outputs for a Node. If the Node does not have delay, this will return 0. For most use cases
// of Nodes with Delay, this will return 1.
func (n *Node) Delay() int {
	if n.delay == nil {
		return 0
	}

	return cap(n.delay)
}

// HasDelay returns whether or not the Node's delay is 0, given by *Node.Delay().
func (n *Node) HasDelay() bool {
	return n.Delay() != 0
}

// Size returns the number of values the Node produces.
func (n *Node) Size() int {
	return n.values.Size()
}

// Dims returns the dimensions of the values that the Node produces. These are directly copied from
// the tensors.Tensor responsible for the holding the Node's values. The returned slice is a copy,
// to allow changes to be made.
func (n *Node) Dims() []int {
	d := make([]int, len(n.values.Dims))
	copy(d, n.values.Dims)
	return d
}

// Shape returns the Tensor responsible for storing the values of the Node. The returned Tensor is
// NOT a copy.
func (n *Node) Shape() tensors.Tensor {
	return n.values
}

// HP returns the values of the given HyperParameter at the current iteration. If an unknown
// HyperParameter is requested, HP will panic with ErrNoHP. This should only happen with custom
// Optimizer types, which can be solved by proper usage of Optimizer.Needs().
func (n *Node) HP(name string) float64 {
	var hp HyperParameter
	if hp = n.hyperParams[name]; hp == nil {
		if hp = n.host.hyperParams[name]; hp == nil {
			panic(ErrNoHP)
		}
	}

	return hp.Value(n.host.longIter)
}

// Value returns the value of the Node at the specified (single-dimensional) index. Value will
// allow panicking with index-out-of-bounds.
func (n *Node) Value(index int) float64 {
	return n.values.Values[index]
}

// PointValue returns the value corresponding to the given point. The point must be valid by the
// dimensions of the Node, else tensors.Tensor.PointValue() will panic. More information about
// possible errors can be found in the documentation for tensors.Interpreter.CheckPoint().
//
// Panics will be either: ErrZeroPoint or type LengthMismatchError or PointOutOfBoundsError.
func (n *Node) PointValue(point []int) float64 {
	return n.values.PointValue(point)
}

// InputValue returns the value of the n'th input to the Node, in a similar fashion to
// *Node.Value(). InputValue will allow panicking with index-out-of-bounds.
//
// NB: For Nodes with many other Nodes as input, this function will be computationally expensive:
// it binary searches all input Nodes.
//
// If InputValue is called on an input Node (which has no input values), it will panic with
// ErrNoInputs
func (n *Node) InputValue(index int) float64 {
	if n.IsInput() {
		panic(ErrNoInputs)
	}

	return n.inputs.value(index)
}

// Delta returns the derivative of the value at the given index w.r.t. the total cost of the
// Network's outputs for the current training sample.
func (n *Node) Delta(index int) float64 {
	return n.deltas[index]
}

// DeltaPoint performs the same operation as *Node.PointValue(), but for the Node's deltas, not its
// values.
func (n *Node) DeltaPoint(point []int) float64 {
	return n.values.Values[n.values.Index(point)]
}

// Input returns the n'th input Node to the given Node. If the Node has no inputs, it will panic
// with ErrNoInputs. Index-out-of-bounds panics are allowed to go through if index < 0 or index >
// n.NumInputNodes().
func (n *Node) Input(index int) *Node {
	if n.IsInput() {
		panic(ErrNoInputs)
	}

	return n.inputs.nodes[index]
}

// InputNodes returns a copy of the set of inputs to the Node. It will return an empty slice if the
// Node has no inputs (is an input Node).
func (n *Node) InputNodes() []*Node {
	ns := make([]*Node, num(n.inputs))
	copy(ns, n.inputs.nodes)
	return ns
}

// NumInputNodes returns the number of Nodes from which the Node recieves input.
func (n *Node) NumInputNodes() int {
	return num(n.inputs)
}

// NumInputs returns the total number of input values to the node.
func (n *Node) NumInputs() int {
	return n.inputs.size()
}

// AllInputs returns a single slice containing a copy of all of the input values to the node, in
// order. Dimensions of those inputs can be obtained by *Node.Input(n).Dims().
func (n *Node) AllInputs() []float64 {
	return n.inputs.getValues(true)
}
