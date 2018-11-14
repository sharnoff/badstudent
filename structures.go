package badstudent

// The main structure that is used to learn to map input to output functions.
// A Network is more of a containing structure than it is actual storage of
// information.
type Network struct {
	inputs, outputs *nodeGroup

	nodesByID []*Node

	err error

	cf CostFunction

	inits []Initializer

	defaultInit Initializer

	hyperParams map[string]HyperParameter

	// used to keep track of the current iteration during training. Also incremented
	// by Correct
	iter int

	// Whether or not there is the possibility of there being a loop in the
	// passage of values from Node to Node
	mayHaveLoop bool

	// Whether or not there are changes to weights that have not been
	// applied yet
	hasSavedChanges bool

	// Whether or not there are any Nodes in the Network with delay.
	// If there are, a different protocol must be followed
	hasDelay bool

	stat status
}

// nodeGroups are a collection of what would instead be individual functions
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

// Node is the fundamental building block with which the network is built.
// Each Node has an Operator, which determines how it changes the values it receives as input
type Node struct {
	// The name that will be used to print this node.
	// Used for unique identification, can be empty
	name string

	// used for order identification of which nodes were added first
	id int

	// used for validation during setup
	host *Network

	// the continuous nodeGroup that the Node belongs to, if there is one
	// otherwise is nil
	group *nodeGroup

	// the root operator
	typ Operator

	// type castings of Operator: (nil if not used)
	adj Adjustable
	el  Elementwise
	lyr Layer

	// nil if adj is nil
	opt Optimizer

	// changes to the weights that have been delayed until the end of the batch
	delayedWeights []float64

	// these are exclusively for the Optimizer
	hyperParams map[string]HyperParameter

	// the values of the node -- essentially its outputs
	values []float64

	// the derivative of each value w.r.t. the total cost of the current
	// training sample
	//
	// if the deltas of this Node should not be calculated, deltas will be nil.
	deltas []float64 // δ

	// whether or not the input deltas need to be calculated. Determined purely
	// by inputs' need to have deltas calculated
	calcInDeltas bool

	// The set of Nodes that the given Node takes input from
	inputs *nodeGroup

	// The set of Nodes that this Node outputs to
	outputs *nodeGroup

	// what index in the network outputs the values of the node start at
	// ex: for the first output node, its 'placeInOutputs' would be 0
	//
	// equal to -1 if not an output
	placeInOutputs int

	tempDelayDeltas []float64

	delay        chan []float64
	delayDeltas  chan []float64
	storedValues [][]float64

	// Whether or not the current task assigned by the network has been
	// completed
	completed bool
}
