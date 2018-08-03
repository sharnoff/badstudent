package badstudent

import (
	"sync"
)

// The main structure that is used to learn to map input to output functions.
// A Network is more of a containing structure than it is actual storage of
// information.
type Network struct {
	inputs, outputs *nodeGroup

	nodesByID   []*Node
	nodesByName map[string]*Node
}

// nodeGroups are a just a collection of what would be individual functions
// because of how different objects handle slices of Nodes
//
// In order to allow nodeGroups to exist, Nodes each have a field: 'group',
// a pointer to the nodeGroup that requires its slice of values to be kept
// in the same location. If there is no nodeGroup relying on a Node's values,
// Node.group will be nil.
//
// nodeGroups default to not being continuous.
// A nodeGroup is created by new(nodeGroup)
type nodeGroup struct {
	// A list of all of the members of the group
	nodes []*Node

	// Only non-nil if in use.
	//
	// If the nodeGroup is continuous -- with values adjacent in memory,
	// 'values' serves as an encapsulating slice that covers the same space as
	// where the individual values of the nodes are stored
	values []float64

	// The sum of the sizes of each Node, up to and including the node at the specified index.
	// For example: index 0 would be equal to the size of the 0th Node; the last index is equal
	// to the size of the entire group.
	sumVals []int
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
	host *Network

	// the continuous nodeGroup that the Node belongs to, if there is one
	// otherwise is nil
	group *nodeGroup

	// handles all of the actual operations from
	typ Operator

	// the values of the node -- essentially its outputs
	values []float64

	// the derivative of each value w.r.t. the total cost
	// of the particular training example
	deltas []float64 // δ

	// separate from 'status.' true iff inputDeltas() have been run on outputs to node
	deltasActuallyCalculated bool

	// The set of Nodes that the given Node takes input from
	inputs *nodeGroup

	// The set of Nodes that this Node outputs to
	outputs *nodeGroup

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
