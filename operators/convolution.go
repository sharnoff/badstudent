package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"math/rand"

	"encoding/json"
	"os"
	// "fmt"
)

// used to provide to the constructor (Convolution())
// has provided public fields to allow for simplified and more easily readible syntax
// These fields can be chaned once given to the constructor
type ConvArgs struct {
	// The Optimizer that will adjust the weights
	Opt Optimizer

	// The dimensions of the output -- usually will be {width, height, depth...}
	// Init() will return error if the product of 'Dims' multiplied together (with Depth, too) doesn't
	// equal the number of values in the layer - *Layer.Size()
	//
	// should not include extra dimensions gained from 'Depth'
	// should also be the same length as 'InputDims'
	// -- remember to include a '1' for dimensions that have been collapsed due to multidimensional filters
	//
	// while this could have been designed to not need 'Dims', it is useful for double-checking
	Dims []int

	// The dimensions of the input
	// Init() checks that the product of 'InputDims' multiplied together equals
	// the total number of input values to the Layer
	InputDims []int

	// The size of the filter in each dimension
	// ex: to have a 3x3x2 filter, this would be {3, 3, 2}
	Filter []int

	// The number of filters made with identical inputs
	// Can be used to expand the dimensionality of the outputs from that of the inputs
	// Recommended to be equal to 1 if 'ParameterSharing' is true, because they would otherwise be the same
	//
	// will default to 1 if ≤ 0
	Depth int

	// The number of inputs per filter, in each dimension
	// If one wanted a filter that alternated each input (width- and height-wise) in 2d,
	// 'Stride' would be {2, 2}
	//
	// can be left nil if filters should be centered at every input
	Stride []int

	// The amount of zero values, or padding, provided on the edges of each dimension
	// Allows filters to be centered closer towards the edges of the inputs
	// to have a ring of padding, one value thick, all around the inputs,
	// this should be {1, 1} in 2d (or {1, 1, 1} in 3d. You get the point.)
	//
	// can be left nil for no padding
	ZeroPadding []int

	// Whether or not all of the filters share the same weights
	// Depth should not be > 1 if this is true
	//
	// currently does nothing - will be enabled in a later update
	ParamSharing bool

	// whether or not each filter has a bias
	Biases bool
}

type convolution struct {
	opt Optimizer

	Outs   *utils.MultiDim
	Ins    *utils.MultiDim // plural of 'in'
	Filter *utils.MultiDim

	Depth int

	Stride      []int
	ZeroPadding []int

	Biases bool

	// currently not enabled
	ParamSharing bool

	// includes biases
	Weights []float64
	Changes []float64
}

const conv_bias_value float64 = 1
const zeroPadding_value float64 = 0

// fills in the initial fields for the Operator,
// does not check for possible errors yet -- that is done with Init()
//
// can be supplied 'nil' if used for Load()
func Convolution(conv *ConvArgs) *convolution {
	if conv == nil {
		return new(convolution)
	}

	c := new(convolution)
	c.opt = conv.Opt

	c.Outs = utils.NewMultiDim(conv.Dims)
	c.Ins = utils.NewMultiDim(conv.InputDims)
	c.Filter = utils.NewMultiDim(conv.Filter)

	c.Depth = conv.Depth

	c.Stride = make([]int, len(conv.Stride))
	copy(c.Stride, conv.Stride)

	c.ZeroPadding = make([]int, len(conv.ZeroPadding))
	copy(c.ZeroPadding, conv.ZeroPadding)

	c.Biases = conv.Biases
	c.ParamSharing = conv.ParamSharing

	// leaves weights unitnitialized
	return c
}

func (c *convolution) Init(l *badstudent.Layer) error {
	numDims := len(c.Outs.Dims)

	// set defauts
	{
		if c.Depth < 1 {
			c.Depth = 1
		}

		if len(c.Stride) == 0 {
			c.Stride = make([]int, numDims)
			for i := range c.Stride {
				c.Stride[i] = 1
			}
		}

		if len(c.ZeroPadding) == 0 {
			c.ZeroPadding = make([]int, numDims) // intentionally left at 0
		}
	}

	// error checking
	{
		if len(c.Outs.Dims) == 0 {
			return errors.Errorf("Can't Init() convolutional layer, len(Dims) == 0")
		}
		if len(c.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() convolutional layer, len(InputDims) == 0")
		}
		if len(c.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() convolutional layer, len(InputDims) != len(Dims). Note: Depth should not be included in dimensions. (len(%v) != len(%v))", c.Ins.Dims, c.Outs.Dims)
		}
		if len(c.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() convolutional layer, len(Filter) != len(Dims) (len(%v) != len(%v))", c.Filter.Dims, c.Outs.Dims)
		}
		if len(c.Stride) != numDims {
			return errors.Errorf("Can't Init() convolutional layer, len(Stride) != len(Dims) (len(%v) != len(%v))", c.Stride, c.Outs.Dims)
		}
		if len(c.ZeroPadding) != numDims {
			return errors.Errorf("Can't Init() convolutional layer, len(ZeroPadding) != len(Dims) (len(%v) != len(%v))", c.ZeroPadding, c.Outs.Dims)
		}

		// check for bad values
		for i, d := range c.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() convolutional layer, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range c.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() convolutional layer, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range c.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() convolutional layer, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range c.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() convolutional layer, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range c.ZeroPadding {
			if d < 0 {
				return errors.Errorf("Can't Init() convolutional layer, ZeroPadding[%d] = %d. Can't have negative padding", i, d)
			}
		}

		// check that the size of the layer is equal to the product of dimensions * depth
		size := 1
		for _, d := range c.Outs.Dims {
			size *= d
		}
		if size*c.Depth != l.Size() {
			return errors.Errorf("Can't Init() convolutional layer, volume of dimensions and depth is not equal to layer size (%d * %d != %d)", size, c.Depth, l.Size())
		}

		// check that the size of the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range c.Ins.Dims {
			inputSize *= d
		}
		if inputSize != l.NumInputs() {
			return errors.Errorf("Can't Init() convolutional layer, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, l.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size + 2 * zero padding) / (stride) + 1
		for i := range c.Outs.Dims {
			numerator := c.Ins.Dims[i] - c.Filter.Dims[i] + (2 * c.ZeroPadding[i])

			if numerator%c.Stride[i] != 0 {
				return errors.Errorf("Can't Init() convolutional layer, stride does not fit Dims[%d] (%d %% %d != 0)", i, numerator, c.Stride[i])
			} else if c.Outs.Dims[i] != numerator/c.Stride[i]+1 {
				return errors.Errorf("Can't Init() convolutional layer, expected Dims[%d] does not match given (%d != %d)", i, numerator/c.Stride[i]+1, c.Outs.Dims[i])
			}
		}
	}

	// allocate weights/biases and changes
	{
		filterSize := 1
		for _, d := range c.Filter.Dims {
			filterSize *= d
		}

		if c.Biases {
			filterSize++
		}

		c.Weights = make([]float64, filterSize*l.Size())
		c.Changes = make([]float64, filterSize*l.Size())

		// initialize weights
		for i := range c.Weights {
			// not a perfect implementation, but it should suffice
			c.Weights[i] = (2*rand.Float64() - 1) / float64(filterSize)
		}
	}

	return nil
}

func (c *convolution) Save(l *badstudent.Layer, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Couldn't save operator: failed to create directory to house save file")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't save operator: failed to create file 'weights.txt'")
	}

	finishedSafely := false
	defer func() {
		if !finishedSafely {
			f.Close()
		}
	}()

	{
		enc := json.NewEncoder(f)
		if err = enc.Encode(c); err != nil {
			return errors.Wrapf(err, "Couldn't save operator: failed to encode JSON to file\n")
		}
		finishedSafely = true
		f.Close()
	}

	if err = c.opt.Save(l, c, dirPath+"/opt"); err != nil {
		return errors.Wrapf(err, "Couldn't save optimizer after saving operator")
	}

	return nil
}

// does not check if the loaded data is intact
// only needs to be provided an empty struct; everything else will be filled in
func (c *convolution) Load(l *badstudent.Layer, dirPath string, aux []interface{}) error {

	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't load operator: could not open file 'weights.txt'")
	}

	finishedSafely := false
	defer func() {
		if !finishedSafely {
			f.Close()
		}
	}()

	{
		dec := json.NewDecoder(f)
		if err = dec.Decode(c); err != nil {
			return errors.Wrapf(err, "Couldn't load operator: failed to decode JSON from file\n")
		}
		finishedSafely = true
		f.Close()
	}

	if err = c.opt.Load(l, c, dirPath+"/opt", aux); err != nil {
		return errors.Wrapf(err, "Couldn't load optimizer after loading operator\n")
	}

	return nil
}

func (c *convolution) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()

	// dimensions, with depth appended to the end
	depthDims := make([]int, len(c.Outs.Dims)+1)
	copy(depthDims, c.Outs.Dims)
	depthDims[len(c.Outs.Dims)] = c.Depth

	// includes biases
	filterSize := 1
	for _, d := range c.Filter.Dims {
		filterSize *= d
	}

	if c.Biases {
		filterSize++
	}

	depthMod := len(values) / c.Depth

	calculateValue := func(index int) {

		// could be optimized later, but will not right now
		depth := index / depthMod
		outIndex := index % depthMod

		out := c.Outs.Point(outIndex)

		var sum float64
		f := make([]int, len(c.Filter.Dims))

		depth_and_out_index := (depth * depthMod) + (outIndex * filterSize)

		filterNoBiases := filterSize
		if c.Biases {
			filterNoBiases--
		}
		for i := 0; i < filterNoBiases; i++ {

			in := make([]int, len(c.Outs.Dims))
			for i := range in {
				in[i] = c.Stride[i]*out[i] + f[i]
			}

			var input float64
			padded := false
			for i := range in {
				if in[i] < 0 || in[i] >= c.Ins.Dims[i] {
					input = zeroPadding_value
					padded = true
				}
			}

			if !padded {
				input = inputs[c.Ins.Index(in)]
			}

			// get weight and add to sum
			if input != 0 {
				// weight := c.Weights[depth_and_out_index + c.Filter.Index(f)]
				// sum += input * weight
				// i := depth_and_out_index + c.Filter.Index(f)
				// if i < 0 || i >= len(c.Weights) {
				// 	fmt.Println(out[len(out)-1], depth_and_out_index, c.Filter.Index(f), i, len(c.Weights))
				// }
				sum += input * c.Weights[depth_and_out_index+c.Filter.Index(f)]
			}

			//f++
			c.Filter.Increment(f)
		}

		if c.Biases {
			sum += bias_value * c.Weights[(outIndex+1)*filterSize-1]
		}

		values[c.Outs.Index(out)] = sum
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

func (c *convolution) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

	filterSize := 1
	for _, d := range c.Filter.Dims {
		filterSize *= d
	}
	if c.Biases {
		filterSize++
	}

	// how much change in index a change in depth should incur
	depthMod := len(c.Weights) / c.Depth // should divide evenly

	sendDelta := func(inputIndex int) {
		input := c.Ins.Point(inputIndex)

		// largest values of 'out'
		out_init := make([]int, len(c.Outs.Dims))

		// smallest values of 'f'
		f_init := make([]int, len(c.Filter.Dims))

		for i := range out_init {
			n := input[i]
			f := c.Filter.Dims[i]
			l := l.NumInputs()
			p := c.ZeroPadding[i]
			s := c.Stride[i]

			r := n + f - l - p
			o := (n + p) / s

			if s*o+f >= l+(2*p) {
				if r%s != 0 {
					out_init[i] = o - (r / s) - 1
				} else {
					out_init[i] = o - (r / s)
				}
			} else {
				out_init[i] = o
			}

			f_init[i] = (n + p) - (s * out_init[i])

			// if the filter isn't in range, then there are no filters that take this
			// value as input. Add nothing to the delta
			if f_init[i] >= f {
				add(inputIndex, 0)
				return
			}
		}

		out := make([]int, len(out_init))
		f := make([]int, len(f_init))
		copy(out, out_init)
		copy(f, f_init)

		var sum float64
		for {
			if f[len(f)-1] >= c.Filter.Dims[len(f)-1] || out[len(out)-1] < 0 {
				break
			}

			// find output value delta
			outIndex := c.Outs.Index(out)
			delta := l.Delta(outIndex)
			// for each depth
			for depth := 0; depth < c.Depth; depth++ {
				weight := c.Weights[(depth*depthMod)+(outIndex*filterSize)+c.Filter.Index(f)]
				sum += weight * delta

			}

			// increment:
			// for each value in f
			for i := range f {
				// increment f[i] (decrement out[i]) by stride[i]
				f[i] += c.Stride[i]
				out[i] -= c.Stride[i]

				if f[i] < c.Filter.Dims[i] && out[i] >= 0 {
					break
				}

				if i == len(f)-1 {
					break
				}

				f[i] = f_init[i]
				out[i] = out_init[i]
			}
		}

		add(inputIndex, sum)
	}

	opsPerThread, threadsPerCPU := 1, 1

	utils.MultiThread(start, end, sendDelta, opsPerThread, threadsPerCPU)

	return nil
}

func (c *convolution) CanBeAdjusted(l *badstudent.Layer) bool {
	return true
}

func (c *convolution) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	inputs := l.CopyOfInputs()

	// either 'c.Weights' or 'c.Changes'
	targets := c.Changes
	if !saveChanges {
		targets = c.Weights
	}

	filterSize := 1
	for _, d := range c.Filter.Dims {
		filterSize *= d
	}
	if c.Biases {
		filterSize++
	}

	depthMod := l.Size() / c.Depth

	grad := func(index int) float64 {
		index %= depthMod

		outIndex := index / filterSize
		index %= filterSize

		// if the given weight is a bias
		if index == filterSize-1 && c.Biases {
			return conv_bias_value * l.Delta(outIndex)
		}

		f := c.Filter.Point(index)

		out := c.Outs.Point(outIndex)
		ins := make([]int, len(out))
		for i := range ins {
			ins[i] = out[i] + f[i]

			// if it's outside the bounds
			if ins[i] < 0 || ins[i] >= c.Ins.Dims[i] {
				return zeroPadding_value * l.Delta(outIndex)
			}
		}

		return inputs[c.Ins.Index(ins)] * l.Delta(outIndex)
	}

	add := func(index int, addend float64) {
		targets[index] += addend
	}

	if err := c.opt.Run(l, l.Size(), grad, add, learningRate); err != nil {
		return errors.Wrapf(err, "Couldn't adjust Operator for layer %v, running optimizer failed\n", l)
	}

	return nil
}

func (c *convolution) AddWeights(l *badstudent.Layer) error {
	for i := range c.Weights {
		c.Weights[i] += c.Changes[i]
	}

	c.Changes = make([]float64, len(c.Weights))

	return nil
}
