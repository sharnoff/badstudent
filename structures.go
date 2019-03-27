package badstudent

import (
	"github.com/sharnoff/tensors"
)

// Network is the main structure that is used to learn to map input to output functions. A Network
// is more of a containing structure than it actual stores information.
type Network struct {
	inputs, outputs *nodeGroup

	// a list of all of the Nodes, stored such that their id is their index in this slice
	nodesByID []*Node

	// whether or not the network should panic when it encounters an error
	panicErrors bool

	err error

	cf CostFunction

	defaultInit Initializer
	defaultOpt  func() Optimizer
	hyperParams map[string]HyperParameter
	pen         Penalty

	// used to keep track of the current iteration during training. Also incremented by Correct
	iter int

	// longIter corresponds to the iteration of the network as a whole, not just within the current
	// training run.
	longIter int

	// Whether or not there is the possibility of there being a loop in the passage of values from
	// Node to Node
	mayHaveLoop bool

	// Whether or not there are changes to weights that have not been applied yet
	hasSavedChanges bool

	// Whether or not there are any Nodes in the Network with delay. If there are, a different
	// protocol must be followed
	hasDelay bool

	stat status
}

// nodeGroups are a collection of what would instead be individual functions because of how
// different objects handle slices of Nodes
//
// In order to allow nodeGroups to exist, Nodes each have a field: 'group', a pointer to the
// nodeGroup that requires its slice of values to be kept in the same location. If there is no
// nodeGroup relying on a Node's values, Node.group will be nil.
//
// nodeGroups default to not being continuous.
// A nodeGroup is created by new(nodeGroup)
type nodeGroup struct {
	// A list of all of the members of the group
	nodes []*Node

	// Only non-nil if in use.
	//
	// If the nodeGroup is continuous -- with values adjacent in memory, 'values' serves as an
	// encapsulating slice that covers the same space as where the individual values of the nodes
	// are stored
	values []float64

	// The sum of the sizes of each Node, up to and including the node at the specified index. For
	// example: index 0 would be equal to the size of the 0th Node; the last index is equal to the
	// size of the entire group.
	sumVals []int
}

// Nodes are the fundamental building blocks with which the Network is built -- they are the nodes
// of the computation graph. Each Node has an Operator that determines how it computes its values
// from those that it receives as input.
type Node struct {
	// The name that will be used to print this node. Completely optional, may be empty.
	name string

	// used for order identification of which nodes were added first
	id int

	// used for validation during setup
	host *Network

	// the continuous nodeGroup that the Node belongs to, if there is one otherwise is nil
	group *nodeGroup

	// The sets of that this Node inputs and outputs from/to
	inputs, outputs *nodeGroup

	// the root operator of the Node
	op Operator

	// type castings of Operator: (nil if not used)
	adj  Adjustable
	elem Elementwise
	lyr  Layer

	opt Optimizer
	pen Penalty

	// changes to the weights that have been delayed until the end of the batch
	delayedWeights []float64

	// these are exclusively for the Optimizer
	hyperParams map[string]HyperParameter

	// the values (essentially outputs) of the Node
	values tensors.Tensor

	// the derivative of each value w.r.t. the total cost of the current training sample. This will
	// be nil (or len=0) if deltas should not be calculated. Deltas are stored in the same ordering
	// as Node.values.
	deltas []float64

	// whether or not the input deltas need to be calculated. Determined purely by inputs' need to
	// have deltas calculated
	calcInDeltas bool

	// outputIndex indicates the index in the Network outputs that this Node's values start at.
	// E.g. for the first output Node, outputIndex equals 0. Non-output Nodes are given values of
	// -1.
	outputIndex int

	tempDelayDeltas []float64

	delay        chan []float64
	delayDeltas  chan []float64
	storedValues [][]float64

	// Whether or not the current task assigned by the Network has been completed. This value takes
	// on different meanings depending on what the Network is doing. It is included as a sort of
	// permenant auxiliary information storage.
	completed bool
}
