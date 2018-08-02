package badstudent

import (
	"sync"
)

// The main structure that is used to learn to map input to output functions.
// A Network is more of a containing structure than it is actual storage of
// information.
type Network struct {
	inputs, outputs nodeGroup

	nodesByID   []*Node
	nodesByName map[string]*Node
}

// nodeGroups are used to put in one place the code that relies on
type nodeGroup struct {
	nodes []*Node

	// only non-nil if in use.
	//
	// if the nodeGroup is continuous -- with values adjacent in memory,
	// 'values' serves as an encapsulating slice that covers the same space as
	// where the individual values of the nodes are stored
	values []float64
}

// The Node is the fundamental building block with which the network is built.
// Each node has a Operator, which determines how it changes the values it recieves as input
type Node struct {
	// The name that will be used to print this node.
	// Used for unique identification, can be empty
	Name string

	// used for order identification of which nodes were added first
	id int

	// used for validation during setup
	hostNetwork *Network

	// handles all of the actual operations from
	typ Operator

	// the values of the node -- essentially its outputs
	values []float64

	// the derivative of each value w.r.t. the total cost
	// of the particular training example
	deltas []float64 // δ

	// separate from 'status.' true iff inputDeltas() have been run on outputs to node
	deltasActuallyCalculated bool

	// the nodes that the given node inputs from
	// could be nil
	inputs []*Node

	// for each node, the sum of the number of its values and those
	// of nodes in previous indexes
	// ex: the index for the last input will have the total
	// number of input values to the node
	numInputs []int

	// the nodes that the given node outputs to
	// can be nil. Can also be non-nil and node is an output
	outputs []*Node

	// whether or not the node's values are part of the set of
	// output values to the network
	isOutput bool

	// what index in the network outputs the values of the node start at
	// ex: for the first output node, its 'placeInOutputs' would be 0
	placeInOutputs int

	// keeps track of what has been calculated or completed for the node
	status status_

	// a lock for 'status'
	statusMux sync.Mutex
}
