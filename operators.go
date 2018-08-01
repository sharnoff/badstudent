package badstudent

// in addition to these functions,
// each operator should be able to provide a way to deserialize its stored data,
// should the network need to be loaded from file
// the signature should be: func(*Node, *os.File) Operator
// for more information, see Load()
type Operator interface {
	// should initialize any weights if used, and return the number of output values from the operation
	// Init() will always be run on an operator before any other method
	// Init() will only be run once
	//
	// can use any available methods on *Node
	// Init(l *Node) (int, error)
	Init(*Node) error

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
	Load(*Node, string, []interface{}) error

	// Should update the values of the node to reflect the inputs and weights (if any)
	// arguments: given node, source slice for the values of that node
	Evaluate(*Node, []float64) error
	// Evaluate(l *Node, values []float64) error

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
	// InputDeltas(l *Node, add func(int, float64), input int) error

	// returns whether or not Adjust() changes the outputs of the Node.
	// generally will be whether or not the Node has weights
	//
	// will be run often, but probably should not change
	// if it changes, there might be unforseen consequences
	CanBeAdjusted(*Node) bool

	// adjusts the weights of the given node, using its deltas
	//
	// args: node to adjust, the learning rate to proivde the optimizer,
	// whether or not the changes from Adjust() should be applied immediately or stored
	Adjust(*Node, float64, bool) error
	// Adjust(l *Node, opt Optimizer, learningRate float64, saveChanges bool) error

	// adds any changes to weights that have been delayed
	//
	// may be called without any changes waiting to happen
	AddWeights(*Node) error
	// AddWeights(l *Node) error
}
