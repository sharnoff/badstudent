package badstudent

import (
	"fmt"
	"github.com/sharnoff/tensors"
)

var defaultOptimizer func() Optimizer
var defaultInitializer Initializer

func SetDefaultOptimizer(f func() Optimizer) {
	defaultOptimizer = f
}

func SetDefaultInitializer(i Initializer) {
	defaultInitializer = i
}

// HandleErrors sets the Network's default response to errors that do not need to be panicked to
// store them until request. This is the recommended state for building around GUIs. Nothing will
// change if HandleErrors is called again. Note: for direct usage of badstudent, this is not
// recommended, as it will decrease the available context to construction errors.
//
// If the provided Network is nil, HandleErrors will panic with ErrNilNet.
func (net *Network) HandleErrors() *Network {
	if net == nil {
		panic(ErrNilNet)
	}

	net.panicErrors = false
	return net
}

// PanicErrors sets the Network's default response to errors to panic them. This allows for more
// more context with direct usage of badstudent. Nothing will change if PanicErrors is called
// again.
//
// If the provided Network is nil, PanicErrors will panic with ErrNilNet.
func (net *Network) PanicErrors() *Network {
	if net == nil {
		panic(ErrNilNet)
	}

	net.panicErrors = true
	return net
}

// initialize performs the small set of actions necessary to set up the network from
// its zero value. It does nothing if it has already been initialized.
func (net *Network) initialize() {
	if net.inputs != nil {
		return
	}

	net.defaultInit = defaultInitializer
	net.hyperParams = make(map[string]HyperParameter)
	net.inputs = new(nodeGroup)
}

// newID adds a Node to the Network's list: net.nodesByID, and retuns the index (id) of the node in
// that lsit.
func (net *Network) newID(n *Node) int {
	id := len(net.nodesByID)
	net.nodesByID = append(net.nodesByID, n)
	return id
}

// AddInput adds an input Node to the Network. Input Nodes cannot have delay.
// AddInput has several conditions in which it'll panic or set net.Error():
// 	(0) net == nil
//	(1) If the network has already been finalized
//	(2) len(dims) == 0
//	(3) dims[i] ≤ 0, for any 0 ≤ 1 < len(dims)
// (0) will panic with ErrNilNet, (1) will panic with ErrNetFinalized, and (2) and (3) are errors
// from tensors.NewTensorSafe: tensors.ErrZeroDims and tensors.DimsValueError, respectively.
func (net *Network) AddInput(dims []int) *Node {
	if net == nil {
		panic(ErrNilNet)
	} else if net.Error() != nil {
		return nil
	} else if net.stat >= finalized {
		panic(ErrNetFinalized)
	}

	net.initialize()

	values, err := tensors.NewTensorSafe(dims)
	if err != nil {
		net.setError(err)
		return nil
	}

	n := &Node{
		host:        net,
		outputs:     new(nodeGroup),
		values:      values,
		outputIndex: -1,

		delay:       make(chan []float64, 0),
		delayDeltas: make(chan []float64, 0),

		hyperParams: make(map[string]HyperParameter),
	}

	n.id = net.newID(n)
	net.inputs.add(n)

	return n
}

// GetShapeError is a wrapper for errors encountered in calls to Operator.OutputShape()
type GetShapeError struct{ error }

func (err GetShapeError) Error() string {
	return "Failed to get output shape: " + err.error.Error()
}

type OperatorFinalizeError struct {
	N *Node
	Err error
}

func (err OperatorFinalizeError) Error() string {
	return fmt.Sprintf("Failed to finalize operator <type: %s> for Node <id: %d>: %s",
		err.N.op.TypeString(), err.N.id, err.Err.Error())
}

// Add adds a new Node to the network with given Operator and inputs. Add returns the Node it
// creates so that it can be chained with method calls. To add an input node, use
// (*Network).AddInput(size int) instead. Also, if net.Error() is not nil, Add will do nothing and
// return nil. If any error is returned or panicked, the Network will not have been changed.
//
// Add has two panic conditions:
//	(0) net == nil or
//	(1) If the Network has already been finalized
// (0) will panic with ErrNilNet and (1) will panic with ErrNetFinalized. Additionally, Add will
// set the Network's stored error and return nil if any of the following are true:
//	(0) op == nil,
//	(1) len(inputs) == 0,
//	(2) any Node in inputs is nil,
//	(3) any Node in inputs belongs to a different Network than 'net',
//	(4) op is not valid (i.e. does not implement Layer or Elementwise), or
//	(5) op.Finalize() returns error
// (0) and (1) give type NilArgError's, (2) gives ErrNilInputNode, (3) gives
// ErrDifferentNetworkInput to BOTH Networks, (4) gives ErrInvalidOperator, and (5) gives type
// OperatorFinalizeError.
func (net *Network) Add(op Operator, inputs ...*Node) *Node {
	if net == nil {
		panic(ErrNilNet)
	} else if net.Error() != nil {
		return nil
	} else if net.stat >= finalized {
		panic(ErrNetFinalized)
	}

	if op == nil {
		net.setError(NilArgError{"Operator"})
		return nil
	} else if len(inputs) == 0 {
		net.setError(NilArgError{"Node 'inputs'"})
		return nil
	}

	net.initialize()

	// that inputs aren't nil, belong to same network
	for _, in := range inputs {
		if in == nil {
			net.setError(ErrNilInputNode)
			return nil
		} else if in.host != net {
			in.host.setError(ErrDifferentNetworkInput)
			net.setError(ErrDifferentNetworkInput)
			return nil
		}
	}

	if !isValid(op) {
		net.setError(ErrInvalidOperator)
		return nil
	}

	shape, err := getShape(op, inputs)
	if err != nil {
		net.setError(err)
		return nil
	}

	// we don't yet know that all of the inputs are safe, but we're going to make the Node anyways
	// to test for certain error cases

	n := &Node{
		host:        net,
		outputIndex: -1,
		outputs:     new(nodeGroup),
		values:      shape,
		hyperParams: make(map[string]HyperParameter),

		delay:       make(chan []float64, 0),
		delayDeltas: make(chan []float64, 0),
	}

	n.inputs = new(nodeGroup)
	n.inputs.add(inputs...)

	n.op = op
	n.lyr, n.elem, n.adj = castAll(op)

	if err := n.op.Finalize(n); err != nil {
		net.setError(OperatorFinalizeError{n, err})
		return nil
	}

	// Now that we know things are safe, let's make the changes that WILL affect other parts of the
	// Network:

	n.id = net.newID(n)

	for _, in := range inputs {
		in.outputs.add(n)
	}

	// mark as incomplete because it has not been initialized
	n.completed = false

	return n
}

// ConcatShape takes multiple tensors and brings them together in the best way that it can. It does
// this simply; for a single input, ConcatShape returns a new tensor with the same dimensions as
// that input, and more inputs are collapsed into a single dimension. A good algorithm for
// combining dimensions has yet to be implemented.
//
// ConcatShape returns type NilArgError if len(inputs) == 0.
func ConcatShape(inputs []tensors.Tensor) (shape tensors.Tensor, err error) {
	if len(inputs) == 0 {
		err = NilArgError{"inputs"}
		return
	} else if len(inputs) == 1 {
		dims := make([]int, len(inputs[0].Dims))
		copy(dims, inputs[0].Dims)
		shape = tensors.NewTensor(dims)
		return
	}

	// a good algorithm for combining dimensions has not been implemented, so we'll just collapse
	// the sizes into one dimension
	size := 0
	for _, in := range inputs {
		size += in.Size()
	}

	return tensors.NewTensor([]int{size}), nil
}

func getShape(op Operator, inputs []*Node) (shape tensors.Tensor, err error) {
	if l, ok := op.(Layer); ok {
		// because of tensor constructors, we know that the dimensions should be fine. If someone's
		// changing it directly, they are allowed to shoot theirself in the foot
		if shape, err = l.OutputShape(inputs); err != nil {
			err = GetShapeError{err}
		}

		return
	} else {
		ls := make([]tensors.Tensor, len(inputs))
		for i := range ls {
			ls[i] = inputs[i].Shape()
		}

		return ConcatShape(ls)
	}
}

// Placeholder returns a placeholder Node that can be used as an input to another Node without its
// (this Node) inputs being set. Usage of placeholders requires declaring the dimensions of the
// Node beforehand. If any error is returned or panicked, the Network will not have been changed.
//
// Placeholder has the same panic conditions as (*Network).Add:
//	(0) If the method is called on a nil Network and
//	(1) If the Network has already been finalized
// (0) will panic with ErrNilNet and (1) will panic with ErrNetFinalized. Additionally, Placeholder
// will set the Network's stored error and return nil if any of the error conditions from
// tensors.NewInterpreterSafe are met. A list is duplicated here for convenience (Note that it may
// not stay up-to-date):
//	(0) If any dimensions are < 0, tensors.DimsValueError;
//	(1) If len(dims) == 0, tensors.ErrZeroDims
func (net *Network) Placeholder(dims []int) *Node {
	// placeholders are marked by setting Node.inputs != nil and Node.op == nil
	// all placeholders have values defined.
	// See (*Network).isPlaceholder()

	if net == nil {
		panic(ErrNilNet)
	} else if net.Error() != nil {
		return nil
	} else if net.stat >= finalized {
		panic(ErrNetFinalized)
	}

	net.initialize()

	values, err := tensors.NewTensorSafe(dims)
	if err != nil {
		net.setError(err)
		return nil
	}

	n := &Node{
		host:    net,
		values:  values,
		inputs:  new(nodeGroup),
		outputs: new(nodeGroup),
	}

	n.id = net.newID(n)

	return n
}

// returns whether or not the Node is a placeholder
func (n *Node) isPlaceholder() bool {
	if n == nil {
		return false
	}

	return n.inputs != nil && n.op == nil
}

// UnequalShapeError documents different shapes of Tensors when Replace()-ing a placeholder Node.
type UnequalShapeError struct {
	Placeholder, Replaced tensors.Tensor
}

func (err UnequalShapeError) Error() string {
	return fmt.Sprintf("Unequal shapes. Placeholder had dims: %v, replaced with: %v", err.Placeholder.Dims, err.Replaced.Dims)
}

// Replace sets the Operator and inputs of a placeholder Node. The dimensions previously indicated
// by calling (*Network).Placeholder must equal those of the tensors.Tensor returned by
// OutputShape. Replace returns the same Node it is called on so as to allow method chaining.
// If any error is returned or panicked, the Node and its Network will not have been changed.
//
// Replace has all of the same error and panic conditions that (*Network).Add() has, excluding
// those that are caused by certain conditions of the network (net == nil || network is finalized).
// Replace has some error conditions of its own:
//	(0) If the provided Node is not a placeholder or
//	(1) If the new dimensions provided by op.OutputShape are not equal to the ones of the
//		placeholder
// (0) will cause a ErrNotPlaceholder, which will be panicked if the Network has been finalized
// (but not if it hasn't) and (1) will cause an error of type UnequalShapeError
func (n *Node) Replace(op Operator, inputs ...*Node) *Node {
	// large sections of this function are borrowed from (*Network).Add()

	if n == nil {
		return nil
	} else if n.host.Error() != nil {
		return nil
	} else if !n.isPlaceholder() {
		if n.host.stat >= finalized {
			panic(ErrNotPlaceholder)
		} else {
			n.host.setError(ErrNotPlaceholder)
			return nil
		}
	}

	if op == nil {
		n.host.setError(NilArgError{"Operator"})
		return nil
	} else if len(inputs) == 0 {
		n.host.setError(NilArgError{"Node 'inputs'"})
		return nil
	}

	// that inputs aren't nil, belong to same network
	for _, in := range inputs {
		if in == nil {
			n.host.setError(ErrNilInputNode)
			return nil
		} else if in.host != n.host {
			in.host.setError(ErrDifferentNetworkInput)
			n.host.setError(ErrDifferentNetworkInput)
			return nil
		}
	}

	if !isValid(op) {
		n.host.setError(ErrInvalidOperator)
		return nil
	}

	shape, err := getShape(op, inputs)
	if err != nil {
		n.host.setError(err)
		return nil
	}

	if !tensors.Equals(n.values.Interpreter, shape.Interpreter) {
		n.host.setError(UnequalShapeError{n.values, shape})
		return nil
	}

	n.inputs.add(inputs...)

	lyr, elem, adj := castAll(op)

	if err := op.Finalize(n); err != nil {
		n.host.setError(OperatorFinalizeError{n, err})
		return nil
	}

	// We now know that error conditions have been avoided, so we'll now make the changes that WILL
	// affect other parts of the network.

	n.op, n.lyr, n.elem, n.adj = op, lyr, elem, adj

	n.outputIndex = -1
	n.values = shape
	n.hyperParams = make(map[string]HyperParameter)

	n.delay = make(chan []float64, 0)
	n.delayDeltas = make(chan []float64, 0)

	for _, in := range inputs {
		in.outputs.add(n)

		if in.id > n.id {
			n.host.mayHaveLoop = true
		}
	}

	// mark as incomplete because it has not been initialized
	n.completed = false

	return n
}

// Finalize completes the structure of the Network by adding a cost function and declaring one or
// more Nodes as the outputs to the Network. If any error is returned, the Network and output Nodes
// will not have been changed. Additionally, if an error has already been encountered earlier by
// the Network, Finalize will do nothing and return that error.
//
// Finalize will not (intentionally) panic, but does have several error conditions (in order of
// precedence):
// 	(0) net == nil:                           ErrNilNet,
//	(1) net.Error() != nil:                   net.Error() -- any previously encountered error,
//	(2) Network has been finalized already:   ErrNetFinalized,
//	(3) Network has no inputs:                ErrNoInputs,
//	(4) len(outputs) == 0:                    ErrNoOutputs,
//	(5) cf == nil: type                       NilArgError,
//  (6) if any output is nil:                 NilArgError,
//  (7) outputs[i] belongs to different net:  ErrDifferentNetworkOutput,
//	(8) if any output is an input:            ErrIsInput,
//	(9) if any output has delay:              ErrOutputHasDelay,
//	(10) if any output Node is repeated:      ErrDuplicateOutput,
//	(11) if any node doesn't affect outputs:  DoesNotAffectOutputsError,
//	(12) if a there is a cycle with 0 delay:  InstantCycleError,
//	(13) if a Node needs optimizer:           ErrNoDefaultOptimizer,
//	(14) if default optimizer returns nil:    NilOptimizerError,
// 	(15) A Node is missing a hpyerparameter:  MissingHyperParamError,
//	(16) if a node is missing an initializer: NoInitializerError,
func (net *Network) Finalize(cf CostFunction, outputs ...*Node) error {
	return net.finalize(false, cf, outputs...)
}

// NoInitializerError results from there not being a default Initializer (package-wide, or for the
// Network), and a certain Node not being initialized. This error will not occur if
// "github.com/sharnoff/badstudent/initializers" is imported, because package initializers sets the
// default package-wide
type NoInitializerError struct {
	N *Node
}

func (err NoInitializerError) Error() string {
	return fmt.Sprintf("Node %v has not been initialized; no default initializer provided.")
}

// finalize is the internal version of Finalize, which is available so that it can be used for
// loading saved Networks.
func (net *Network) finalize(isLoading bool, cf CostFunction, outputs ...*Node) error {
	if net == nil {
		return ErrNilNet
	} else if net.Error() != nil {
		return net.Error()
	} else if net.stat >= finalized {
		return ErrNetFinalized
	} else if num(net.inputs) == 0 {
		return ErrNoInputs
	} else if len(outputs) == 0 {
		return ErrNoOutputs
	} else if cf == nil {
		return NilArgError{"Provided CostFunction"}
	}

	// this ensures that all given output Nodes remain unchanged
	var allGood = false
	defer func() {
		if !allGood {
			for _, out := range outputs {
				if out != nil {
					out.outputIndex = -1
				}
			}
		}
	}()

	index := 0
	for i, out := range outputs {
		if out == nil {
			return NilArgError{fmt.Sprintf("Output Node #%d", i)}
		} else if out.host != net {
			return ErrDifferentNetworkOutput
		} else if out.IsInput() {
			return ErrIsInput
		} else if out.HasDelay() {
			return ErrOutputHasDelay
		}

		// check there are no duplicates
		// this doesn't need to be fast; it's only run once
		for o := i + 1; o < len(outputs); o++ {
			if out == outputs[o] {
				return ErrDuplicateOutput
			}
		}

		out.outputIndex = index
		index += out.Size()
	}

	net.outputs = new(nodeGroup)
	net.outputs.add(outputs...)

	// returns DoesNotAffectOutputsError or InstantCycleError
	if err := net.checkGraph(); err != nil {
		return err
	}

	for _, n := range net.nodesByID {
		if n.opt == nil && n.adj != nil {
			if defaultOptimizer == nil {
				return ErrNoDefaultOptimizer
			}

			opt := defaultOptimizer()
			if opt == nil {
				return NilOptimizerError{n}
			}

			n.opt = opt
		}

		if err := n.checkHPs(); err != nil {
			return err
		}
	}

	// Past this point, no errors should be encountered
	allGood = true

	net.cf = cf

	if net.defaultInit == nil {
		net.defaultInit = defaultInitializer
	}

	// If any nodes have not been initialized and are adjustable, do so, so long as we aren't
	// loading the network from a save.
	// Also, trim slices that have more space allocated than necessary
	for _, n := range net.nodesByID {
		// trim to slightly reduce memory usage. Realistically, it doesn't make much of a
		// difference either way, but it helps my OCD
		n.outputs.trim()
		n.inputs.trim()

		// if it needs initializing
		if n.adj != nil && !isLoading {

			if net.defaultInit != nil {
				net.defaultInit.Set(n, n.adj.Weights())
			} else if defaultInitializer != nil {
				defaultInitializer.Set(n, n.adj.Weights())
			} else {
				return NoInitializerError{n}
			}

		}
	}

	// allocate single slices for inputs and outputs
	net.inputs.makeContinuous()
	net.outputs.makeContinuous()

	net.stat = finalized

	return nil
}

// NilOptimizerError results from and documents Adjustable Nodes that have been given nil
// Optimizers from a default Optimizer. This error (mostly) should not occur.
type NilOptimizerError struct {
	N *Node
}

func (err NilOptimizerError) Error() string {
	return fmt.Sprintf("Default Optimizer provided to Node %v is nil - Operator is Adjustable", err.N)
}

// MissingHyperParamError documents errors ocurring from missing hyperparameters that have not been
// supplied to a certain Node.
type MissingHyperParamError struct {
	N *Node

	// Name is the name of the missing HyperParameter
	Name string
}

func (err MissingHyperParamError) Error() string {
	return fmt.Sprintf("Node %v mising HyperParameter with name %q", err.N, err.Name)
}

// checkHPs checks that all hyperparameter requirements are met and that the Node
// has an Optimzier
func (n *Node) checkHPs() error {
	if n.adj == nil {
		return nil
	}

	needs := n.opt.Needs()
	for _, s := range needs {
		if _, ok := n.hyperParams[s]; !ok {
			if hp, ok := n.host.hyperParams[s]; ok {
				n.hyperParams[s] = hp
				continue
			}

			return MissingHyperParamError{n, s}
		}
	}

	return nil
}

// AddHP adds the given HyperParameter to the set of HyperParameters that will be supplied by the
// Node to its Optimizer and Operator.
//
// AddHP will panic if the Network has already been finalized. It will also return NilArgError if
// the HyperParameter is nil, and ErrHPNameTaken if the name has already been registered to the
// Node.
func (n *Node) AddHP(name string, hp HyperParameter) *Node {
	if n == nil || n.host.err != nil {
		return n
	} else if n.host.stat >= finalized {
		panic(ErrNetFinalized)
	}

	if hp == nil {
		n.host.setError(NilArgError{"HyperParameter"})
		return n
	} else if n.hyperParams[name] != nil {
		n.host.setError(ErrHPNameTaken)
		return n
	}

	n.hyperParams[name] = hp
	return n
}

// ReplaceHP replaces the HyperParameter of the given name for the given Node. It is different from
// AddHP in that it can only be run after the Network has been finalized.
//
// ReplaceHP has multiple error conditions (and none from which it'll panic):
//	(0) If n == nil,
//	(1) If the Network has not been finalized,
//	(2) If hp == nil,
//	(3) If the Node has no HyperParameter with for the given name.
// (0) and (2) return type NilArgErrors, (1) returns
func (n *Node) ReplaceHP(name string, hp HyperParameter) error {
	if n == nil {
		return NilArgError{"Node"}
	} else if n.host.stat < finalized {
		return ErrNetNotFinalized
	} else if hp == nil {
		return NilArgError{"HyperParameter"}
	} else if _, has := n.hyperParams[name]; !has {
		return ErrNoHPToReplace
	}

	n.hyperParams[name] = hp
	return nil
}

// AddHP effectively performs the same operation as *Node.AddHP, but adds the HyperParameter to a
// Network-wide list that Nodes will default to if that HyperParameter has not already been set.
//
// *Network.AddHP has the same error conditions as *Node.AddHP.
func (net *Network) AddHP(name string, hp HyperParameter) *Network {
	if net.Error() != nil {
		return net
	} else if net.stat >= finalized {
		panic(ErrNetFinalized)
	}

	if hp == nil {
		net.setError(NilArgError{"HyperParameter"})
		return net
	} else if net.hyperParams[name] != nil {
		net.setError(ErrHPNameTaken)
		return net
	}

	net.hyperParams[name] = hp
	return net
}

// ReplaceHP effectively performs the same operation as *Node.ReplaceHP, but on the Network's list
// of HyperParameters. It has the same error conditions as *Node.ReplaceHP.
func (net *Network) ReplaceHP(name string, hp HyperParameter) error {
	if net == nil {
		return NilArgError{"Network"}
	} else if net.stat < finalized {
		return ErrNetNotFinalized
	} else if hp == nil {
		return NilArgError{"HyperParameter"}
	} else if _, has := net.hyperParams[name]; !has {
		return ErrNoHPToReplace
	}

	net.hyperParams[name] = hp
	return nil
}

// Opt sets the Optimizer of the Node. If this is not called, the Optimizer will be set from the
// default.
//
// Opt will panic with ErrNetFinalized if the network has been fully set-up, and will set the host
// Network's stored error to type NilArgError if the given Optimizer is nil
func (n *Node) Opt(opt Optimizer) *Node {
	if n == nil {
		return n
	} else if n.host.stat >= finalized {
		panic(ErrNetFinalized)
	} else if n.host.Error() != nil {
		return n
	} else if opt == nil {
		n.host.setError(NilArgError{"Optimizer"})
		return n
	}

	n.opt = opt
	return n
}

// SetPenalty sets the penalty on the weights of the node -- This is completely optional.
// SetPenalty returns the Node it is called on so that methods can be chained if necessary. If the
// Node's Operator is not Adjustable (i.e. if it doesn't have weights) then SetPenalty will have no
// measurable effect.
//
// SetPenalty will panic with ErrNetFinalized if the Network has been finalized, and will set the
// Network's error to type NilArgError if the given Penalty is nil.
func (n *Node) SetPenalty(p Penalty) *Node {
	if n == nil || n.host.Error() != nil {
		return n
	} else if n.host.stat >= finalized {
		panic(ErrNetFinalized)
	} else if p == nil {
		n.host.setError(NilArgError{"Penalty"})
	}

	n.pen = p
	return n
}

// SetPenalty sets the default penalty of all Nodes in the Network. Only Nodes with Adjustable
// Operators (those with weights) will have their penalty set.
//
// SetPenalty will panic with ErrNilNet if the Network is nil, ErrNetFinalized if the network has
// been finalized, and will set the Network's error to type NilArgError if the given Penalty is
// nil.
func (net *Network) SetPenalty(p Penalty) *Network {
	if net == nil {
		panic(ErrNilNet)
	} else if net.Error() != nil {
		return net
	} else if net.stat >= finalized {
		panic(ErrNetFinalized)
	} else if p == nil {
		net.setError(NilArgError{"Penalty"})
	}

	net.pen = p
	return net
}

// SetDelay sets the number of time-steps in between calculation of the Node's values and those
// calculated values becoming inputs for other Nodes. This will usually be set to 1.
//
// SetDelay has several error/panic conditions:
//	(0) If the host Network has been finalized,
//	(1) If 'n' is a placeholder Node,
//	(2) If 'n' is an input Node, or
//	(3) If 'delay' < 0.
// (0) will cause a panic with ErrNetFinalized. The other three will set the Network's error to
// ErrDelayPlaceholder, ErrDelayInput, and ErrInvalidDelay, respectively.
func (n *Node) SetDelay(delay int) *Node {
	if n == nil {
		return n
	} else if n.host.stat >= finalized {
		panic(ErrNetFinalized)
	} else if n.host.Error() != nil {
		return n
	}

	if delay == n.Delay() {
		return n
	} else if n.isPlaceholder() {
		n.host.setError(ErrDelayPlaceholder)
		return n
	} else if n.IsInput() {
		n.host.setError(ErrDelayInput)
		return n
	} else if delay < 0 {
		n.host.setError(ErrInvalidDelay)
		return n
	}

	if delay == 0 {
		n.delay, n.delayDeltas = nil, nil

		// update whether or not the network has delay.
		n.host.hasDelay = false
		for _, t := range n.host.nodesByID {
			if t.HasDelay() {
				n.host.hasDelay = true
				break
			}
		}

		return n
	}

	n.delay = make(chan []float64, delay)
	n.delayDeltas = make(chan []float64, delay)

	// fill up the delays needed
	for i := 0; i < delay; i++ {
		n.delay <- make([]float64, len(n.values.Values))
		n.delayDeltas <- make([]float64, len(n.values.Values))
	}

	n.host.hasDelay = true

	return n
}

// Init initializes the weights of the Node. This is only required if the Node's Operator is
// Adjustable. If not provided, the default Initializer (first from the Network, DefaultInit(),
// then from the package-wide, SetDefaultInitializer()) will be used instead, assuming they are
// set.
//
// Init should be prioritized last in method chains, if possible.
//
// Init will panic with ErrNetFinalized if the Network has already been finalized, and will set the
// Network's error to type NilArgError if the provided Initializer is nil.
func (n *Node) Init(i Initializer) *Node {
	if n == nil || n.host.err != nil {
		return n
	} else if n.host.stat >= finalized {
		panic(ErrNetFinalized)
	} else if i == nil {
		n.host.setError(NilArgError{"Initializer"})
		return n
	}

	i.Set(n, n.adj.Weights())
	return n
}

// DefaultInit sets the Initializers of all Nodes in the Network that do not (or
// will not) have Initializers directly set.
//
// DefaultInit will panic with ErrNetFinalized if the Network has already been finalized, and will
// set the Network's error to type NilArgError if the provided Initializer is nil.
func (net *Network) DefaultInit(i Initializer) *Network {
	if net.stat >= finalized {
		panic(ErrNetFinalized)
	} else if i == nil {
		net.setError(NilArgError{"Initializer"})
		return net
	}

	net.defaultInit = i
	return net
}

// SetName sets a name for the node to provide instead for calls to *Node.String(). Names are not
// guaranteed to be unique, and can be set to any string. Empty strings will be interpreted as
// being without names. Names can be retrieved by *Node.Name()
func (n *Node) SetName(name string) *Node {
	if n == nil {
		return n
	}

	n.name = name
	return n
}
