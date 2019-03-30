// Package badstudent provides a lower-level framework for maintaining various types of neural
// networks. It also allows high levels of customizability in all parts of the system.
//
// Creating Networks
//
// The center of all training is the Network, initialized by:
//
//		net := new(bs.Network)
//
// For brevity, badstudent is abbreviated 'bs'.
//
// Networks consist of graphs of Nodes, which are analagous to the typical layer or activation
// function. Each Node has an Operator, which determines its values and the backpropagation through
// it. For the Operators with weights (such as layers of neurons), additional Optimziers are
// required. (More information later) All Operators can be found in the subpackage "operators", all
// Optimizers in "optimziers", and so forth, for other types.
//
// The standard procedure for adding Nodes to the Network is:
//
//		in := net.AddInput([]int{inputSize})
//		hl := net.Add(operators.Neurons(size), in)
//		hl.Opt(optimizers.SGD()).AddHP("learning-rate", hyperparams.Constant(0.1))
//
//		if net.Error() != nil {
//			return net.Error()
//		}
//
// All Nodes have a shape (in terms of their dimensions), whether explicitly or implicitly. In the
// case of input Nodes, the only argument to the Network method AddInput() is the shape of the
// Node. For most Nodes, dimensions will be implied (especially in the case of convolutional
// layers).
//
// Optimizers are added on a per-Node basis, but defaults can be set at the package level with
// SetDefaultOptimizer(). Additionally, Nodes whose Operators have weights must have Initializers,
// which can be set by default. The default Initializer will set weights to be uniformly random,
// independent of size, provided that the subpackage "initializers" is imported.
//
// *Network.Add() works excellently for feed-forward networks, but other tools are required to add
// loops:
//
//		l := net.Placeholder([]int{loopSize})
//		// ... add other Nodes, some of which take 'l' as input
//		l.Replace(operators.Neurons(), l).SetDelay(1)
//
// Here, we establish a placeholder first, so that it may be used as an input to itself. It could,
// of course, be used as an input to others before being Replace'd or could have a delay of more
// than 1. In order for the Network to properly finalize, all loops within the architecture of the
// Network must have at least one Node with Delay.
//
// The network can be finished by providing a cost function:
//
//		err := net.Finalize(costfuncs.MSE(), hl)
//		if err != nil {
//			return err
//		}
//
// operators, optimizers, hyperparams, initializers, and costfuncs are - of course - subpackages
// of badstudent.
//
// Training and Testing
//
// Training is mildly cumbersome, with the type TrainArgs used as a proxy for the type of optional
// arguments that are available in other languages (such as Python). Training and Testing are all
// done with the custom type Datum, which contains two slices of float64 for inputs and correct
// outputs for the Network.
//
// All training is done with the function Train:
//
//		func (net *Network) Train(args TrainArgs)
//
// Testing can be done both during training (see TrainArgs) and through a separate
// function, Test:
//
//		func (net *Network) Test(data DataSupplier, isCorrect func([]float64, []float64) bool) (float64, float64, error)
//
// While this seems overly complicated, it is just a subsection of fields found in TrainArgs. More
// information can be found there.
//
// Saving and Loading
//
// Writing Networks to files is quite simple. The function signature is:
//
//		func (net *Network) Save(dirPath string, overwrite bool) error
//
// dirPath is the path to create the directory (nothing should be there), or to overwrite if you so
// desire. Loading is equally simple, with:
//
//		func (net *Network) Load(dirPath string) (*Network, error)
package badstudent