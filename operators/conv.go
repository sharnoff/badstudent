package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"github.com/sharnoff/tensors"
)

type convConstructor struct {
	dims      []int
	inputDims []int
	filter    []int
}

type conv struct {
	*convConstructor

	Outs *utils.MultiDim
	Ins  *utils.MultiDim
	// Filt short for Filter
	Filt *utils.MultiDim

	// Dep short for depth
	Dep         int
	ShareParams bool

	// Str is short for stride
	Str          []int
	Padding      []int
	PaddingValue float64

	// always either 0 or 1. It is represented as an integer to make the math easier and to reduce
	// the number of necessary conditionals
	NumBiases int

	// the value multiplied by bias
	Bias float64

	// weights are stored by the output value they correspond to, with the smaller filter indexes
	// within. Biases are appended to the end.
	Ws []float64
}

// const default_numBiases int = 1
const default_depth int = 1
const default_paramShare bool = false

// Conv returns a typical convolutional function with weights, available for any number of
// dimensions, which implements badstudent.Operator.
//
// Conv does not return a completed Operator, however. The methods InputDims and Filter must be
// called in order to provide enough information to Finalize the Operator. Other methods can be
// called to to further customize it -- they return *conv and do not check for errors, so they can
// be chained.
func Conv() *conv {
	c := new(conv)
	c.Dep = default_depth
	c.Bias = defaultValue["conv-bias"]
	c.NumBiases = default_numBiases
	c.ShareParams = default_paramShare
	c.PaddingValue = defaultValue["conv-padding"]
	c.convConstructor = new(convConstructor)
	return c
}

// ***************************************************
// Customization functions
// ***************************************************

// Dims sets the output dimensions of the convolution Operator. It does not check that the
// dimensions are valid until it is Finalized. Dims will panic if called after the Operator has
// been finalized. Depth does not need to be included as a dimension in Dims. If Depth > 1, Dims
// will have an appended dimension of value equal to Depth.
//
// Dims is optional, as it can be calculated from other information, but it can also be provided.
//
// If any dimensions collapse to a size of 1, they must still be included.
func (c *conv) Dims(dims ...int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.dims = dims
	return c
}

// InputDims sets the input dimensions of the convolutional Operator, for internal use. This is
// REQUIRED unless the Operator is being loaded. InputDims will panic if called after the Operator
// has been Finalized.
func (c *conv) InputDims(dims ...int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.inputDims = dims
	return c
}

// Filter sets the size of the filter in each dimension. This is REQUIRED unless the Operator is
// being loaded. Filter will panic if called after the Operator has been Finalized.
func (c *conv) Filter(dims ...int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.filter = dims
	return c
}

// Stride sets the space between centers of filter regions. Stride defaults to the same size as the
// filters. Stride will panic if called after the Operator has been Finalized.
//
// At finalization, the convolutional Operator will return error if any dimensions of Stride are
// larger than Filter.
//
// Stride is optional.
func (c *conv) Stride(dims ...int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.Str = dims
	return c
}

// Pad sets the amount of padding on both ends for each dimension. Pad defaults to none, if not
// provided. Pad will panic if called after the Operator has been Finalized.
func (c *conv) Pad(dims ...int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.Padding = dims
	return c
}

// PadValue sets the value of the padding around the input. The default can be set by
// SetDefault("conv-padding").
func (c *conv) PadValue(v float64) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.PaddingValue = v
	return c
}

// Depth sets the number of 'copies' of the output that are going to be made. Different parameters
// are used for each depth. Depth defaults to 1, and will cause Finalize to error if less than 1.
func (c *conv) Depth(d int) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.Dep = d
	return c
}

// ParamSharing sets whether or not the parameters within a certain Depth are shared for all copies
// of the filter. ParamSharing defaults to false.
func (c *conv) ParamSharing(share bool) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.ShareParams = share
	return c
}

// WithBiases adds bias inputs and parameters to each filter. Convolutional operators default to
// having biases.
func (c *conv) WithBiases() *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.NumBiases = 1
	return c
}

// NoBiases removes the bias inputs and paramaters from each filter. Convolutional operators
// default to having biases.
func (c *conv) NoBiases() *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.NumBiases = 0
	return c
}

// BiasValue sets the value multiplied by the biases. The default value can be set by
// SetDefault("conv-bias")
func (c *conv) BiasValue(b float64) *conv {
	if c.convConstructor == nil {
		panic("convolutional Operator has already been finalized")
	}

	c.Bias = b
	return c
}

// ***************************************************
// Helper Functions
// ***************************************************

// MustSize calls Size, but panics if it encounters an error.
func (c *conv) MustSize() int {
	size, err := c.Size()
	if err != nil {
		panic(err.Error())
	}

	return size
}

// Size returns the expected size of the convolutional Operator, before it has been finalized. If
// the configuration is invalid, it will return error, just as Finalize would.
func (c *conv) Size() (int, error) {
	if c.Outs != nil {
		return c.Outs.Size() * c.Dep, nil
	}

	if c.inputDims == nil {
		return 0, errors.Errorf("InputDims has not been set")
	} else if c.filter == nil {
		return 0, errors.Errorf("Filter has not been set")
	}

	dimList := [][]int{c.inputDims, c.dims, c.filter, c.Padding}
	names := []string{"InputDims", "Dims", "Filter", "Stride", "Padding"} // only used in case of error
	for i := range dimList {
		if dimList[i] == nil {
			continue
		}

		if len(dimList[i]) == 0 {
			return 0, errors.Errorf("%s has been set with length zero", names[i])
		} else if len(dimList[i]) != len(c.inputDims) {
			return 0, errors.Errorf("%s has different length to InputDims (%d != %d)", names[i], len(dimList[i]), len(c.inputDims))
		}

		for d := range dimList[i] {
			if dimList[i][d] < 1 {
				return 0, errors.Errorf("%s[%d] = %d, must be ≥ 1", names[i], d, dimList[i][d])
			}
		}
	}

	if c.Dep < 1 {
		return 0, errors.Errorf("Depth is < 1 (%d)")
	}

	if c.Str == nil {
		c.Str = c.filter
	} else {
		for d := range c.Str {
			if c.Str[d] > c.filter[d] {
				return 0, errors.Errorf("Filter[%d] is less than than Stride[%d] (%d > %d)", d, d, c.filter[d], c.Str[d])
			}
		}
	}

	if c.Padding == nil {
		c.Padding = make([]int, len(c.inputDims))
	}

	if c.dims != nil { // if everything is filled in, check whether or not it works
		for i := range c.inputDims {
			in := c.inputDims[i] + 2*c.Padding[i]
			d := c.dims[i]
			f := c.filter[i]
			s := c.Str[i]

			if (in+s-f)%s != 0 {
				return 0, errors.Errorf("Dimenision #%d does not divide evenly: (InputDim + 2*Padding + Stride - Filter) %% (Stride) != 0 "+
					"((%d + 2*%d + %d - %d)%%%d = %d)", i, c.inputDims[i], c.Padding[i], s, f, s, (in+s-f)%s)
			} else if (in+s-f)/s != d {
				return 0, errors.Errorf("Dimension #%d does not produce desired output (InputDim + 2*Padding + Stride - Filter) / (Stride) != OutputDim "+
					"((%d + 2*%d + %d - %d) / (%d) != %d", i, c.inputDims[i], c.Padding[i], s, f, s, d)
			}
		}
	} else { // fill in
		c.dims = make([]int, len(c.inputDims))

		for i := range c.inputDims {
			in := c.inputDims[i] + 2*c.Padding[i]
			f := c.filter[i]
			s := c.Str[i]

			if (in+s-f)%s != 0 {
				return 0, errors.Errorf("Dimenision #%d does not divide evenly: (InputDim + 2*Padding + Stride - Filter) %% (Stride) != 0 "+
					"((%d + 2*%d + %d - %d)%%%d != 0)", i, c.inputDims[i], c.Padding[i], s, f, s)
			}

			c.dims[i] = (in + s - f) / s
		}
	}

	c.Outs = utils.NewMultiDim(c.dims)
	c.Ins = utils.NewMultiDim(c.inputDims)
	c.Filt = utils.NewMultiDim(c.filter)

	c.convConstructor = nil

	return c.Outs.Size() * c.Dep, nil
}

// returns whether or not the point in the inputs (set of inputs plus padding) is outside the
// boundaries of the actual inputs
func (c *conv) isPadding(point []int) bool {
	p := make([]int, len(point))
	copy(p, point)
	mapSub(p, c.Padding)

	// if any are less than 0, it's padding
	for i := range p {
		if p[i] < 0 || p[i] >= c.Outs.Dim(i) {
			return true
		}
	}

	return false
}

// out_p does not include depth
func (c *conv) inputsTo(out_p []int) []int {
	// here, underscores are used as a suffix to indicate the type of the variable. For example,
	// x_i would be an index with name 'x', and x_p would be an n-dimensional point with name 'x'.

	// topleft is the top-left base point for the filter, including padding
	topLeft_p := mapMult(out_p, c.Str)

	// the list of input indexes to be supplied to c
	inList := make([]int, c.Filt.Size())

	// the point in relation to the top left of the filter
	fMod := make([]int, len(c.Str)) // stride is the easiest way to get the number of dimensions
	for i := 0; i < c.Filt.Size(); i++ {

		// in is the current input that we're looking at, including padding
		var in_p []int
		{
			in_p = make([]int, len(topLeft_p))
			copy(in_p, topLeft_p)
			mapAdd(in_p, fMod)
		}

		if c.isPadding(in_p) {
			// -1 indicates that it is out of range of the inputs -- it's padding
			inList[i] = -1
		} else {
			inList[i] = c.Ins.Index(mapSub(in_p, c.Padding))
		}

		if i+1 < c.Filt.Size() {
			c.Filt.Increment(fMod)
		}
	}

	return inList
}

// returns the index in c.Ws
// mod is the modification from the filter
func (c *conv) weight(out, mod []int, depth int) float64 {
	filterSize := c.Filt.Size() + c.NumBiases
	if c.ShareParams {
		return c.Ws[depth*filterSize+c.Filt.Index(mod)]
	} else {
		return c.Ws[depth*c.Outs.Size()*filterSize+
			c.Outs.Index(out)*filterSize+
			c.Filt.Index(mod)]
	}
}

// returns the index in c.Ws
func (c *conv) bias(out []int, depth int) float64 {
	filterSize := c.Filt.Size() + c.NumBiases
	if c.ShareParams {
		return c.Ws[depth*filterSize+c.Filt.Size()]
	} else {
		return c.Ws[depth*c.Outs.Size()*filterSize+
			c.Outs.Index(out)*filterSize+
			c.Filt.Size()]
	}
}

// ***************************************************
// Interface implementation
// ***************************************************

func (t *conv) TypeString() string {
	return "conv"
}

func (t *conv) Finalize(n *bs.Node) error {
	size, err := t.Size()
	if err != nil {
		return err
	}

	if t.Ins.Size() != n.NumInputs() {
		return errors.Errorf("Mismatch between expected number of inputs and actual (%d != %d)", t.Ins.Size(), n.NumInputs())
	}

	var wLen int
	if !t.ShareParams {
		wLen = (t.Filt.Size() + t.NumBiases) * size
	} else {
		wLen = (t.Filt.Size() + t.NumBiases) * t.Dep
	}

	if t.Ws == nil {
		t.Ws = make([]float64, wLen)
	} else if len(t.Ws) != wLen {
		return errors.Errorf("Number of saved weights not equal to expected number (%d != %d)", len(t.Ws), wLen)
	}

	return nil
}

func (t *conv) Get() interface{} {
	return *t
}

func (t *conv) Blank() interface{} {
	return t
}

func (t *conv) OutputShape(ins []*bs.Node) (tensors.Tensor, error) {
	_, err := t.Size()
	if err != nil {
		return tensors.Tensor{}, err
	}

	return tensors.NewTensor(t.Outs.Dims), nil
}

func (t *conv) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()

	f := func(v int) {
		depth := v / t.Outs.Size()

		out_p := t.Outs.Point(v % t.Outs.Size())
		ins := t.inputsTo(out_p)
		mod := make([]int, len(t.Str))
		var sum float64
		for _, in := range ins {
			if in == -1 {
				sum += t.PaddingValue * t.weight(out_p, mod, depth)
			} else {
				sum += inputs[in] * t.weight(out_p, mod, depth)
			}

			t.Filt.Increment(mod)
		}

		if t.NumBiases != 0 {
			sum += t.Bias * t.bias(out_p, depth)
		}

		values[v] = sum
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t *conv) InputDeltas(n *bs.Node) []float64 {
	atoms := make([]uint64, n.NumInputs())

	f := func(out int) {
		depth := out / t.Outs.Size()
		out_p := t.Outs.Point(out % t.Outs.Size())
		ins := t.inputsTo(out_p)
		mod := make([]int, len(t.Str))
		for _, in := range ins {
			if in != -1 {
				atomAdd(&atoms[in], n.Delta(out)*t.weight(out_p, mod, depth))
			}
			t.Filt.Increment(mod)
		}
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)

	return uint64ToFloat64(atoms)
}

func (t *conv) Grad(n *bs.Node, index int) float64 {
	filterSize := (t.Filt.Size() + t.NumBiases)

	mod := index % filterSize
	index /= filterSize

	out := index % t.Outs.Size()
	depth := index / t.Outs.Size()

	if mod == t.Filt.Size() { // if it's a bias
		return t.Bias * n.Delta(out)
	} else {
		in_p := mapAdd(mapMult(t.Outs.Point(out), t.Str), t.Filt.Point(mod))

		if t.isPadding(in_p) {
			return t.PaddingValue * n.Delta(out+depth*t.Outs.Size())
		} else {
			return n.InputValue(t.Ins.Index(mapSub(in_p, t.Padding))) * n.Delta(out+depth*t.Outs.Size())
		}
	}
}

func (t *conv) Weights() []float64 {
	return t.Ws
}
