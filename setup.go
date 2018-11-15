package badstudent

import (
	"github.com/pkg/errors"
)

var defaultOptimizer func() Optimizer
var defaultInitializer Initializer

func SetDefaultOptimizer(f func() Optimizer) {
	defaultOptimizer = f
}

func SetDefaultInitializer(i Initializer) {
	defaultInitializer = i
}

// initialize performs the small set of actions necessary to set up the network from
// its zero value.
func (net *Network) initialize() {
	if net.inputs != nil {
		return
	}

	net.defaultInit = defaultInitializer
	net.hyperParams = make(map[string]HyperParameter)
	net.inputs = new(nodeGroup)
}

// Add adds a new Node to the Network, with given name, size, inputs, and Operator.
// If no inputs are given, the Node will be an input Node, and its size added to the
// number of inputs. Input Nodes must not have an Operator.
//
// The name of the Node is not necessary, but is useful for facilitating debugging,
// if necessary.
//
// If this process results in an error, the host network will not have changed, and
// the returned Node will be nil
//
// Add creates and replaces a placeholder Node, calling *Network.Placeholder and
// *Node.Replace.
func (net *Network) Add(name string, typ Operator, size int, inputs ...*Node) *Node {
	n := net.Placeholder(name, size)
	if net.Error() != nil {
		return nil
	}

	n.Replace(typ, inputs...)
	if net.Error() != nil {
		net.nodesByID = net.nodesByID[:len(net.nodesByID)-1]
		return nil
	}

	return n
}

// Placeholder returns a placeholder Node that can be used as an input to another
// Node without its [this Node] inputs being set. All placeholders must be replaced
// before the Network can be Finalized.
//
// Placeholder will panic if provided Network is nil.
func (net *Network) Placeholder(name string, size int) *Node {
	if net == nil {
		panic("Provided network is nil")
	} else if net.Error() != nil {
		return nil
	} else if net.stat > initialized {
		net.err = errors.Errorf("Network has already finished construction")
		return nil
	}

	net.initialize()

	if size < 1 {
		net.err = errors.Errorf("Node must have size >= 1 (%d)", size)
		return nil
	}

	n := &Node{
		name:           name,
		host:           net,
		id:             len(net.nodesByID),
		placeInOutputs: -1,
		outputs:        new(nodeGroup),
		values:         make([]float64, size),
		hyperParams:    make(map[string]HyperParameter),
	}

	net.inits = append(net.inits, nil)
	net.nodesByID = append(net.nodesByID, n)
	return n
}

// Replace sets the Operator and inputs of a placeholder Node. The network must
// still be in construction, and all inputs must belong to the same Network.
//
// Replace will panic if provided Node is nil.
func (n *Node) Replace(typ Operator, inputs ...*Node) *Node {
	if n == nil {
		panic("Provided Node is nil")
	} else if !n.IsPlaceholder() {
		n.host.err = errors.Errorf("Provided Node %v is not a placeholder", n)
		return nil
	}
	// now that we know it's a placeholder, the network also must have finished
	// construction
	if typ == nil && len(inputs) != 0 {
		n.host.err = errors.Errorf("Operator is nil and Node is not an input")
		return nil
	} else if len(inputs) == 0 && typ != nil {
		n.host.err = errors.Errorf("Node is input but Operator is not nil")
		return nil
	} else if typ != nil && !isValid(typ) {
		n.host.err = errors.Errorf("Operator is not valid; must be either Layer or Elementwise")
		return nil
	}

	for i, in := range inputs {
		if in == nil {
			n.host.err = errors.Errorf("Input %d to Node is nil", i, n)
			return nil
		} else if in.host != n.host {
			in.host.err = errors.Errorf("Attempted replacement of node from another Network")
			n.host.err = errors.Errorf("Input %d (%v) to Node does not belong to the same Network", i, in)
			return nil
		}

		if in.id > n.id {
			n.host.mayHaveLoop = true
		}
	}

	// Everything has been cleared
	n.typ = typ
	n.lyr, n.el, n.adj = castAll(typ)

	n.inputs = new(nodeGroup)
	n.inputs.add(inputs...)

	if n.el != nil && n.Size() != n.NumInputs() {
		n.host.err = errors.Errorf("Element-wise Node %v size (%d) unequal to input size (%d)", n.Size(), n.NumInputs())
	}

	// initialize at 0, can be updated later
	n.delay = make(chan []float64, 0)
	n.delayDeltas = make(chan []float64, 0)

	if len(inputs) == 0 {
		n.host.inputs.add(n)
	}

	for _, in := range inputs {
		in.outputs.add(n)
	}

	// mark as completed for *Node.IsPlaceholder()
	n.completed = true

	return n
}

// Finalizes the structure of the Network.
//
// No outputs can be inputs
// All Nodes must affect the outputs
//
// If an error is returned, the Network has remained unchanged
func (net *Network) Finalize(cf CostFunction, outputs ...*Node) error {
	return net.finalize(false, cf, outputs...)
}

func (net *Network) finalize(isLoading bool, cf CostFunction, outputs ...*Node) error {
	// Note: A possible optimization would be to check for cases where nodegroups
	// are the same and use the same references.

	if net.Error() != nil {
		return net.Error()
	} else if net.stat >= finalized {
		return errors.Errorf("Network has already been finalized")
	} else if num(net.inputs) == 0 {
		return errors.Errorf("Network has no inputs")
	} else if len(outputs) == 0 {
		return errors.Errorf("No outputs given")
	} else if cf == nil {
		return errors.Errorf("No CostFunction given")
	}

	// Remove complpetion marks by Replace()
	net.resetCompletion()

	var allGood = false
	defer func() {
		if !allGood {
			for _, out := range outputs {
				if out != nil {
					out.placeInOutputs = -1
				}
			}
		}
	}()

	numOutVals := 0
	for i, out := range outputs {
		if out == nil {
			return errors.Errorf("Output Node #%d is nil", i)
		} else if out.host != net {
			return errors.Errorf("Output Node #%d (%v) does not belong to this Network", i, out)
		} else if out.IsInput() {
			return errors.Errorf("Output Node #%d (%v) is also an input", i, out)
		} else if out.Delay() != 0 {
			return errors.Errorf("Output Node #%d (%v) illegally has delay", i, out)
		}

		// check there are no duplicates.
		// this doesn't need to be fast; it's only run once
		for o := i + 1; o < len(outputs); o++ {
			if out == outputs[o] {
				return errors.Errorf("Output #%d is also #%d", i, o)
			}
		}

		out.placeInOutputs = numOutVals
		numOutVals += out.Size()
	}

	net.outputs = new(nodeGroup)
	net.outputs.add(outputs...)

	if err := net.checkGraph(); err != nil {
		return err
	}

	for _, n := range net.nodesByID {
		if n.typ != nil {
			if err := n.typ.Finalize(n); err != nil {
				return errors.Wrapf(err, "Failed to finalize node %v (id %d)\n", n, n.id)
			}
		}

		if n.opt == nil && n.adj != nil {
			n.opt = defaultOptimizer()
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

	// Slightly reduce memory usage, mostly just to help my OCD
	for _, n := range net.nodesByID {
		n.outputs.trim()
		n.inputs.trim()
		if n.adj != nil && !isLoading { // if loading, don't overwrite weights
			if net.inits[n.id] != nil {
				net.inits[n.id].Set(n, n.adj.Weights())
			} else if net.defaultInit != nil {
				net.defaultInit.Set(n, n.adj.Weights())
			} else {
				return errors.Errorf("Node %v missing initializer, default has not been set.", n)
			}
		}
	}

	// allocate single slices for inputs and outputs
	net.inputs.makeContinuous()
	net.outputs.makeContinuous()

	net.stat = finalized

	return nil
}

// checkHPs checks that all hyperparameter requirements are met and that the Node
// has an Optimzier
func (n *Node) checkHPs() error {
	if n.adj == nil {
		return nil
	}

	if n.opt == nil {
		return errors.Errorf("Node %v (id %d) is Adjustable but has no Optimzier", n, n.id)
	}

	needs := n.opt.Needs()
	for _, s := range needs {
		if _, ok := n.hyperParams[s]; !ok {
			if hp, ok := n.host.hyperParams[s]; ok {
				n.hyperParams[s] = hp
				continue
			}

			return errors.Errorf("Node %v (id %d) missing hyperparameter %q", n, n.id, s)
		}
	}

	return nil
}

// AddHP adds the given HyperParameter to the set of HyperParameters that will be
// supplied by the Node to its Optimizer and Operator. If an error is encountered,
// it will be reported via Network-wide error, and AddHP will simply do nothing if
// an eror is already present when called. If AddHP is called after setup, it will
// panic.
//
// AddHP returns the Node it is called on so that it can be stacked, e.g.:
// 	n.AddHP("1", hp1).AddHP("2", hp2).AddHP("3", hp3)
// However, the returned Node points to the same location -- It is the same Node, so
// does not need to be additionally stored.
func (n *Node) AddHP(name string, hp HyperParameter) *Node {
	if n == nil { // could be from an earlier error
		return n
	} else if n.host.stat >= finalized {
		panic("Cannot add HyperParameter, Network has been finalized")
	} else if n.host.err != nil {
		return n
	}

	if n.hyperParams[name] != nil {
		n.host.err = errors.Errorf("Cannot add HyperParameter (name: %q) to Node %v, name is already taken", name, n)
		return n
	} else if hp == nil {
		n.host.err = errors.Errorf("Cannot add HyperParameter (name: %q) to Node %v, HyperParameter is nil", name, n)
		return n
	}

	n.hyperParams[name] = hp
	return n
}

// ReplaceHP replaces the HyperParameter with the given name. It is different from
// AddHP in that it can only be run after the Network has been finalized.
//
// ReplaceHP will return error only if the Node does not already have that
// HyperParameter, if the HyperParameter is nil, if the Network has not been
// finalized, or if the provided Node is nil.
func (n *Node) ReplaceHP(name string, hp HyperParameter) error {
	if n == nil {
		return errors.Errorf("Node is nil")
	} else if n.host.stat < finalized {
		return errors.Errorf("Network has not been finalized")
	} else if hp == nil {
		return errors.Errorf("HyperParameter is nil")
	} else if _, has := n.hyperParams[name]; !has {
		return errors.Errorf("No HyperParameter to replace (would be adding new)")
	}

	n.hyperParams[name] = hp
	return nil
}

// AddHP adds the given HyperParameter to all Nodes within the Network
func (net *Network) AddHP(name string, hp HyperParameter) *Network {
	if net.stat >= finalized {
		panic("Cannot add HyperParameters, Network has been finalized")
	} else if net.err != nil {
		return net
	}

	if net.hyperParams[name] != nil {
		net.err = errors.Errorf("Cannot add HyperParameter (name: %q), name is already taken", name)
	} else if hp == nil {
		net.err = errors.Errorf("Cannot add HyperParameter (name: %q), is nil", name)
	} else {
		net.hyperParams[name] = hp
	}

	return net
}

// ReplaceHP applies *Node.ReplaceHP to all Nodes with a HyperParameter already
// there to replace.
func (net *Network) ReplaceHP(name string, hp HyperParameter) error {
	for _, n := range net.nodesByID {
		if _, has := n.hyperParams[name]; has {
			if err := n.ReplaceHP(name, hp); err != nil {
				return errors.Wrapf(err, "Failed to replace HyperParameter for Node %v (id #%d)\n", n, n.id)
			}
		}
	}

	return nil
}

// Opt sets the Optimizer of the Node. This can only be run during setup, and will
// panic if called after setup.
//
// Opt returns the Node it is called on so that it can be stacked, in a similar
// fashion to AddHP
func (n *Node) Opt(opt Optimizer) *Node {
	if n == nil {
		return n
	} else if n.host.stat >= finalized {
		panic("Cannot set Optimizer, Network has been finalized.")
	} else if n.host.err != nil {
		return n
	}

	if opt == nil {
		n.host.err = errors.Errorf("Cannot set Optimizer of Node %v, Optimizer is nil", n)
	}

	n.opt = opt
	return n
}

// SetDelay sets the amount of delay in the Node
//
// Constraints:
//
// The delay cannot be set after the network has been Finalize'd. Input Nodes cannot
// have delay because their values are set directly. Output Nodes also cannot have
// delay. Placeholder Nodes cannot have their delay set because their inputs are not
// yet known.
//
// If any error is encountered, it will be reported through (*Network).Error.
// SetDelay will panic if it is called after the network has been Finalized.
func (n *Node) SetDelay(delay int) *Node {
	if n == nil {
		return n
	} else if n.host.stat >= finalized {
		panic("Cannot set Delay, Network has been finalized")
	} else if n.host.err != nil {
		return n
	}

	if delay == n.Delay() {
		return n
	}

	if n.IsPlaceholder() {
		n.host.err = errors.Errorf("Cannot set Delay of Node %v, is a placeholder", n)
		return n
	} else if n.IsInput() {
		n.host.err = errors.Errorf("Cannot set Delay of Node %v, is an input Node", n)
		return n
	}

	n.delay = make(chan []float64, delay)
	n.delayDeltas = make(chan []float64, delay)
	for i := 0; i < delay; i++ {
		n.delay <- make([]float64, len(n.values))
		n.delayDeltas <- make([]float64, len(n.values))
	}

	if delay != 0 {
		n.host.hasDelay = true
	} else {
		// update whether or not the network has delay
		n.host.hasDelay = false
		for _, t := range n.host.nodesByID {
			if t.HasDelay() {
				n.host.hasDelay = true
				break
			}
		}
	}

	return nil
}

// Init sets the initializer of the Node. If not provided, the default Initializer
// will be used.
func (n *Node) Init(i Initializer) *Node {
	if n == nil {
		return n
	} else if n.host.stat >= finalized {
		panic("Cannot set initializer, Network has been finalized")
	} else if n.host.err != nil {
		return n
	}

	n.host.inits[n.id] = i
	return n
}

// DefaultInit sets the Initializers of all Nodes in the Network that do not (or
// will not) have Initializers directly set.
func (net *Network) DefaultInit(i Initializer) *Network {
	net.defaultInit = i
	return net
}
