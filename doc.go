// Package badstudent provides a lower-level framework for maintaining
// various types of neural networks. It also allows high levels of
// customizability in all parts of the system.
//
// Overview
//
// The primary type for general operations is the Network, constructed by:
//
//		net := new(bs.Network)
//
// For reference, badstudent is typically abbreviated 'bs'
//
// Networks are graphs of Nodes, from inputs to outputs, with each Node
// holding its own Operator, which determines the values of the Node.
// Operators can be made by the user for unique purposes, but common ones
// can be found in subpackage badstudent/operators. Some Operators futher
// use a type defined in badstudent/operators and implemented in a
// subpackage: Optimizer.
//
// Setup
//
// There are two ways to add a Node to the Network, the first being:
//
//		in, err := net.Add("inputs", nil, <input size>, bs.NoDelay)
//		if err != nil {
//			return err
//		}
//
//		hl, err := net.Add("hidden layer neurons", operators.Neurons(optimizers.GradientDescent()), <hidden layer size>, bs.NoDelay)
//		if err != nil {
//			return err
//		}
//
// Nodes with no given inputs (and no Operator) will be assigned to be
// network inputs.
//
// This works excellently for feed-forward networks, but using just this
// function fails to allow for loops. TO do that, there are two companion
// functions, (*Network).Placeholder(…) and (*Node).Replace(…).
// Placeholder does what it says; it creates a Node that can be used as an
// input to other Nodes, without requiring the place-holder to have its
// own inputs. Add calls Placeholder(), then Replace() to set the Node.
//
// To construct a loop taking input from the previous hidden layer:
//
//		loop, err := net.Placeholder("loop", <loop size>)
//		if err != nil {
//			return err
//		}
//
//		if err = loop.Replace(operators.Neurons(optimizers.GradientDescent()), bs.Delay(1), loop) {
//			return err
//		}
//
// Note: Loops within the architecture must have a delay of at least 1.
// Also Note: Nodes can input from themselves when they are replaced, as
// evidenced above.
//
// To finish constructing the architecture of the network:
//
//		if err = net.SetOutputs(<list of Nodes>); err != nil {
//			return err
//		}
//
// Training and Testing
//
// This functionality is, again, fully available to be customized by the
// user, but mostly provided by the package itself.
//
// Training and testing are done with a custom type, Datum, which just
// contains two slices of float64 for inputs and outputs to the network.
// All training (and testing for progress) is done with the function
//
//		func (net *Network) Train(args TrainArgs) {…}
//
// TrainArgs is a struct meant to fill the role of allowing optional
// arguments to the training of the network. TrainArgs is fully
// documented, but there are some types that it contains that will be
// explained here.
//
//		type DataSupplier interface {…}
//
// DataSupplier is the way that all networks are supplied with data during
// training and testing. If you are working with large datasets, it may be
// advantageous to implement this yourself. Otherwise, Data() will convert
// [][][]float64 or []Datum to DataSuppliers, so that you don't need to.
//
// One important field in TrainArgs is 'Results', a channel of type
// Result. This is fairly self-explanatory, but should be briefely
// examined before use.
//
// The fields
//
//		ShouldTest func(int) bool
//		SendStatus func(int) bool
//
// can be filled by helper functions such as Every() and EndEvery();
//
//		RunCondition func(int, float64) bool
//
// can be filled by TrainUntil(); and
//
//		LearningRate func(int, float64) float64
//
// can be filled by ConstantRate()
//
// Saving and Loading
//
// Saving a network is fairly simple. The function signature is
//
//		func (net *Network) Save(dirPath string, overwrite bool) error {…}
//
// There should be nothing at 'dirPath' unless 'overwrite' is true.
//
// Loading a network is slightly more complex. Because there's no way to
// directly save the type of an interface, they must be provided when the
// network is loaded. This is done with a map of string to Operator, the
// Node's name to its Operator.
//
//		func Load(dirPath string, types map[string]Operator, aux map[string][]interface{}) error {…}
package badstudent