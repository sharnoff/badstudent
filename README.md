[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-go.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/60-percent-of-the-time-works-every-time.svg)](https://forthebadge.com)

# badstudent

### What is it?

badstudent is a general-purpose machine-learning library written in and for go. It's not designed to be super
effective - I'm just using it as a way to learn more about the lower-level parts of neural networks. It
excels at providing flexibility to the user - including multiple-time-step delay and free-form layer architecture.

This framework has been a solo project, but it's perfectly usable for other people. (There are better
libraries out there, though)

### How do I use it?

badstudent is based on Nodes and Operators. Nodes are essentially the wrappers around the 'user supplied'
Operators. The Operators are only practically supplied by other parts of the library (badstudent/operators).
Some of these Operators (such as a layer of neurons) require additional Optimizers, which also can either be
supplied by the user or the library (badstudent/operators/optimizers).

#### Setup:

Setting up a network is fairly simple. Because there is no separate type of network for recurrent models,
there is no initialization function. Instead, just create a new network by:
```
import (
    bs "github.com/sharnoff/badstudent"
)

net := new(bs.Network)
```
Of course, it's possible to supply a `*Network` with giving the address of an empty struct, but it's simpler to
just construct a reference.

Adding layers to the network is simple, too. `(*Network).Add(…)` is the usual way to add to the architecture of
the network.
```
(*bs.Network).Add(name string, typ bs.Operator, size, delay int, inputs ...*bs.Node) (*bs.Node, error)
```
`delay` will usually be zero - it is the number of time-steps in-between a change in inputs and that corresponding
change in outputs. `Add()` doesn't allow for loops by itself (because inputs must already be initialized), so you
can make 'placeholder' Nodes in the network and replace them later. Placeholders can be used as inputs, and must
be replaced later. Nodes created with no given inputs will be used as input nodes, with indexes in the order they
were created. Additionally, no Operator should be given for input nodes.
```
(*bs.Network).Placeholder(name string, size int) (*bs.Node, error)
(*bs.Node).Replace(typ Operator, delay int, inputs ...*bs.Node) (error)
```
After the architecture of the network has been completed, simply call `(*Network).SetOutputs(…)` to finish up.
```
(*bs.Network).SetOutputs(outputs ...*bs.Node) error
```

###### An example with exclusive-or (xor):
```
// additional imports
import (
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/operators/optimizers"
)

l, err := net.Add("inputs", nil, 2, 0)
if err != nil {
	panic(err.Error())
}

if l, err = net.Add("hidden layer neurons", operators.Neurons(optimizers.GradientDescent()), 3, 0, l); err != nil {
	panic(err.Error())
}

if l, err = net.Add("hidden layer logistic", operators.Logistic(), 3, 0, l); err != nil {
	panic(err.Error())
}

if l, err = net.Add("output neurons", operators.Neurons(optimizers.GradientDescent()), 1, 0, l) {
	panic(err.Error())
}

if l, err = net.Add("output logistic", operators.Logistic(), 1, 0, l) {
	panic(err.Error())
}

if err = net.SetOutputs(l); err != nil {
	panic(err.Error())
}
```
While this example is a little wordy, this can also be done without error checking, which collapses the code to be
more compact.

#### Training and Testing:

All of the code surrounding this can be found in traintest.go

Training is fairly simple. The `Network` method `Train(…)` takes one argument: `args TrainArgs`. `TrainArgs` is a
type defined to allow many optional parameters to training. There are no pre-stored datasets, but instead there
are easy ways to supply your own. The type `DataSupplier` serves to supply training and testing data to the
network. `Data(…)` allows conversion to this type from slices or from our internal type, `Datum`. Results are sent
back through a user-provided channel of `Result`, which contains information about testing and the status of the
training.

Datum is a simple wrapper for training or testing data that can be used to supply it to the network.
```
type Datum struct {
	Inputs  []float64
	Outputs []float64
}
```

###### Annotated `TrainArgs` Struct Members:
* `TrainData DataSupplier`: This is the source of data used for training the network. `Data(…)` works to fill this
field.
* `TestData DataSupplier`: The source of testing data. Can be left nil if `ShouldTest` is nil.
* `ShouldTest func(int) bool`: Whether or not the network should test before the given iteration. `Every(…)` and
`EndEvery(…)` work to satisfy this. **Recurrent models should ensure testing in-between sets**
* `SendStatus func(int) bool`: Whether or not to send back the status of training before the given iteration
* `RunCondition func(int, float64) bool`: Whether or not the network should continue training on the current
iteration and given the previous cost. `TrainUntil(…)` satisfies this.
* `LearningRate func(int, float64) float64`: Determines the learning-rate for each iteration of training,
additionally given the previous cost. `ConstantRate(…)` satisfies this.
* `IsCorrect func([]float64, []float64)`: Whether or not the network outputs are correct, given the expected
outputs. `CorrectRound` and `CorrectHighest` both satisfy this. Will default to `CorrectRound`.
* `CostFunc CostFunction`: WYSIWYG. Defaults to `SquaredError(…)`.
* `Results chan Result`: Where the results are sent.
* `Err *error`: Because this function runs in another thread, this is where any error is returned

`Result` is the wrapper for information sent back about training.
```
type Result struct {
	Iteration int

	// Average cost, from CostFunc
	Cost float64

	// The fraction correct, from IsCorrect() from TrainArgs. Ranges from 0 → 1
	Correct float64

	// The result is either from a test or a status update
	IsTest bool
}
```

#### Saving and Loading:
###### Saving:
Saving a network is simple. Just `(*Network).Save(…)` with a specified path:
```
(*bs.Network).Save(dirPath string, overwrite bool) error
```
\*Note: `dirPath` should provide a path including the save directory, but that directory should not exist unless
`overwrite` is `true`.

###### Loading:
Loading is slightly more complex. Because there is no way to load an from file without knowing what
type it is, you (the user) must supply `Load()` with the Operators for each Node.
```
Load(dirPath string, types map[string]Operator, aux map[string][]interface{}) (*bs.Network, error)
```
The string as the key for `types` is the name of the Node. `aux` is the auxiliary information that gets passed on
to the Operators (and is currently unnecessary).
