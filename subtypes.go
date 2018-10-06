package badstudent

// Operator is an interface for defining layers and activation functions
type Operator interface {
	// should initialize any weights if used, and return the number of output values from the operation
	// Init() will always be run on an operator before any other method
	// Init() will only be run once
	//
	// can use any available methods on *Node
	// Init(l *Node) (int, error)
	Init(*Node) error

	// TypeString returns the string corresponding to the type of the Operator.
	// For example: the Operator "Identity" should return "identity", or something
	// to that effect.
	TypeString() string

	// given a path to a directory (and the name of it, without a '/' at the end)
	// should store enough information to recreate the Operator from file, should
	// the need arise
	//
	// the directory will not be created, used, or altered by the library itself
	Save(*Node, string) error

	// given a path to a directory (and the name of it, without a '/' a the end)
	// should use the information already in the directory to recreate this Operator
	// from a file
	// should produce the same result as Init(). The provided node will be at the
	// same stage as Init() in its construction
	//
	// the directory will not be created, used, or altered by the library itself
	Load(*Node, string) error

	// Should update the values of the node to reflect the inputs and weights (if any)
	// arguments: given node, source slice for the values of that node
	Evaluate(*Node, []float64) error

	// Should return the nth value of the Operator to reflect the inputs
	//
	// If anything goes wrong, panic.
	Value(*Node, int) float64

	// should add to the deltas of the given range of inputs how each of those values affects the
	// total error through the values of the host node
	//
	// arguments: given node, a way to add to the deltas of the given input,
	// starting index of those input values, ending index of those input values
	// more details on add:
	// adds the given 'float64' to the deltas of the given input at index 'int'
	// returns error if out of bounds
	//
	// Is likely to be called in parallel for multiple inputs,
	// so this needs to be thread-safe.
	InputDeltas(*Node, func(int, float64), int, int) error

	// returns whether or not Adjust() changes the outputs of the Node.
	// generally will be whether or not the Node has weights
	//
	// will be run once, during setup. This should not change.
	CanBeAdjusted(*Node) bool

	// returns whether or not the the values of this Node are needed in order to
	// calculate its deltas or adjust weights
	NeedsValues(*Node) bool

	// returns whether or not the input values to this Node are needed in order to
	// calculate its deltas or adjust weights
	//
	// will likely be run multiple times during setup -- the result should remain the same
	NeedsInputs(*Node) bool

	// adjusts the weights of the given node, using its deltas
	//
	// args: node to adjust, the learning rate to proivde the optimizer,
	// whether or not the changes from Adjust() should be applied immediately or stored
	Adjust(*Node, float64, bool) error

	// adds any changes to weights that have been delayed
	//
	// may be called without any changes waiting to happen
	AddWeights(*Node) error
}

// Optimizer is an interface
type Optimizer interface {
	// Run is called to suggest changes to each weight, given:
	// number of weights, gradient at weight, function to add to weights,
	// and a learning-rate
	Run(*Node, int, func(int) float64, func(int, float64), float64) error

	// TypeString returns the string corresponding to the type of the Optimizer.
	// For example: the Optimizer "Adam" should return "adam", or something
	// to that effect.
	TypeString() string

	Save(*Node, string) error
	Load(*Node, string) error
}

// Initializer dictates how the weights in an Operator will be set, given
// a blank slice to hold weights
type Initializer func(*Node, []float64)
