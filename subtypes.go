package badstudent

// Storable is an optional additional interface for Operators and Optimizers.
// It allows saving and loading with files for types that implement it, and is
// a required part of some additional interfaces
type Storable interface {
	// Save saves the object, given a path to a directory, with no
	// appended backslash.
	Save(n *Node, dirPath string) error

	// Load loads the object, given a path to a directory, with no appended
	// backslash.
	//
	// For Operators, Loading will take place before compilation.
	Load(dirPath string) error
}

/*
// delayed for later

// Uses is a list of the necessary values for the process of backpropagation
// and weight updating.
//
// Uses is primarily an optimization tool, allowing the model to not store
// some values, and instead just calculate them once. Improper use of Uses will
// result in sub-par performance
type Uses struct {
	InputValues, OwnValues bool
}
*/

// Operator is the basic interface for defining Layers and Activation
// functions. All Operators must also be either a Layer or an Elementwise in
// order to function at runtime. Additionally, some Operators may be
// Adjustable -- and should be -- if they have weights. If an Operator with
// weights is not Adjustable, it will not be adjusted by badstudent internals.
type Operator interface {
	// TypeString returns a constant, type-unique string corresponding to the type
	// of the Operator. It is only called during saving and loading. It is not a
	// part of Storable because even Operators that do not store additional
	// information must have some method of conveying their type.
	//
	// For example: the Identity Operator returns "identity"
	TypeString() string

	// Finalize initializes needed components of the Operator, checking that all
	// numbers -- mainly inputs and values -- are valid, and sufficiently
	// prepares for full use.
	Finalize(*Node) error
}

// Layer is an interface on top of Operator that is for definining
// non-elementwise Operators. Note: Some activation functions will fall under
// this category (ex: Softmax)
type Layer interface {
	Operator

	// Evaluate calculates the values of the Operator and sets the values of
	// the provided slice to those values.
	Evaluate(n *Node, values []float64)

	// InputDeltas returns the component of the 'deltas' (i.e. derivative
	// w.r.t. cost) of the inputs that is a result of the Operator called.
	InputDeltas(*Node) []float64
}

// Elementwise is a simpler interface to implement for Operators that have forward
// and backward propagation passes that are each completely elementwise. Most
// activation functions will fall under this category, but not all. Some layers
// might also fall under this category.
type Elementwise interface {
	Operator

	// Value applies the elementwise function to the given value. This will be
	// run in parallel. Value is also given the index of the value, in case to
	// account for functions that will differ based on the index (like PReLU).
	Value(v float64, index int) float64

	// Deriv returns the derivative of the elementwise function for the
	// particular index given. This will be run in parallel. The value can be
	// obtained by *Node.Value()
	Deriv(n *Node, index int) float64
}

// Adjustable is an extension on top of Operator for those that have adjustable
// weights
type Adjustable interface {
	Operator
	Storable

	// Grad returns the gradient of the weight specified by the given index. This is
	// determined, in part, using the deltas (derviative w.r.t. total cost) of the
	// Node.
	Grad(n *Node, index int) float64

	// Weights returns the weights of an Adjustable Operator so that they can
	// be set by initialization or updated by Optimizers. The number of weights
	// given must remain constant throughout runtime.
	//
	// Weights should return the actual stored weights so that they can be
	// efficiently set and updated. It is recommended that Weights be simple to
	// allow for compile-time in-lining and prevent overhead.
	Weights() []float64
}

func isValid(o Operator) bool {
	if _, ok := o.(Layer); ok {
		return true
	} else if _, ok := o.(Elementwise); ok {
		return true
	}

	return false
}

func castAll(o Operator) (l Layer, el Elementwise, adj Adjustable) {
	l, _ = o.(Layer)
	el, _ = o.(Elementwise)
	adj, _ = o.(Adjustable)
	return
}

// Optimizer is an interface things that updates the weights of Adjustable
// Operators
type Optimizer interface {
	// TypeString returns a constant, unique string corresponding to the type of the
	// Optimizer. It is only called during saving and loading.
	//
	// For example: the Optimizer "Adam" returns "adam"
	TypeString() string

	// Run is called to add changes to each weight in the Operator in order to
	// minimize the cost function. Run is also given a slice to make the changes,
	// which can be either the original weights or temporary storage for batch-wide
	// changes.
	Run(n *Node, adj Adjustable, changes []float64)

	// Needs returns a list of names of HyperParameters that are needed to run the
	// Optimizer. These can be obtained at each iteration from *Node.HP()
	Needs() []string
}

// CostFunction is the interface defined for allowing measures of model performance,
// attached to the Network at finalization. Like Operators, it must be registered
// before it can be loaded.
type CostFunction interface {
	// TypeString returns a constant, type-unique string corresponding to the type
	// of the CostFunction. It is only called during saving and loading.
	//
	// For example: the Constant HyperParameter returns "constant"
	TypeString() string

	// Cost returns the cost (negative performance) given actual values, target
	// values. (also referred to as yHat and y, respectively) The lengths will be
	// guaranteed to be equal.
	Cost(outs, targets []float64) float64

	// Deriv returns the derivatives of each value w.r.t. the total cost, given the
	// actual values and target values.
	Derivs(outs, targets []float64) []float64
}

// HyperParameter is the method for providing user-defined values to Optimizers.
// Like Operators, they must be registered before they can be loaded.
//
// Note that the registered type of the HyperParameter is independent of the value
// that it provides. E.g. a Constant() hyperparameter may provide "learning-rate".
type HyperParameter interface {
	// TypeString returns a constant, type-unique string corresponding to the type
	// of the HyperParameter. It is only called during saving and loading.
	//
	// For example: the Constant HyperParameter returns "constant"
	TypeString() string

	// Value returns the desired value of the HyperParameter at a given iteration.
	// Value will sometimes be called multiple times for the same iteration, and it
	// should give the same result.
	Value(iter int) float64
}

// Initializer sets the initial weights in an Adjustable Operator.
type Initializer interface {
	Set(n *Node, weights []float64)
}
