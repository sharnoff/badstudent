package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"

	"encoding/json"
	"fmt"
	"os"
)

type PoolArgs struct {
	// The dimensions of the output -- usually will be {width, height, depth...}
	// Init() will return error if the product of 'Dims' multiplied together doesn't
	// equal the number of values in the node - *Node.Size()
	//
	// should be the same length as 'InputDims'
	//
	// while this could have been designed to not need 'Dims', it is useful for double-checking
	Dims []int

	// The dimensions of the input
	// Init() checks that the product of 'InputDims' multiplied together equals
	// the total number of input values to the Node
	InputDims []int

	// The size of the filter in each dimension
	// ex: to have a 3x3x2 filter, this would be {3, 3, 2}
	//
	// this is the size of the area that will be pooled
	Filter []int

	// The number of inputs per filter, in each dimension
	// If one wanted a filter that alternated each input (width- and height-wise) in 2d,
	// 'Stride' would be {2, 2}
	//
	// can be left nil if filters should be centered at every input
	Stride []int
}

type pool struct {
	Outs   *utils.MultiDim
	Ins    *utils.MultiDim
	Filter *utils.MultiDim

	Stride []int
}

type avgPool pool

type maxPool struct {
	pool

	// the index (in inputs) of the highest value
	switches []int
}

func constructPool(args *PoolArgs) pool {
	p := pool{
		Outs:   utils.NewMultiDim(args.Dims),
		Ins:    utils.NewMultiDim(args.InputDims),
		Filter: utils.NewMultiDim(args.Filter),
	}

	p.Stride = make([]int, len(args.Stride))
	copy(p.Stride, args.Stride)

	return p
}

// can be supplied 'nil' if being used to load from file
func AvgPool(args *PoolArgs) *avgPool {
	ap := avgPool(constructPool(args))
	return &ap
}

// can be supplied 'nil' if being used to load from file
func MaxPool(args *PoolArgs) *maxPool {
	return &maxPool{constructPool(args), nil}
}

// ***************************************************
// AvgPool:
// ***************************************************

func (a *avgPool) Init(n *bs.Node) error {

	numDims := len(a.Outs.Dims)

	// set defaults
	if a.Stride == nil {
		a.Stride = make([]int, numDims)
		for i := range a.Stride {
			a.Stride[i] = 1
		}
	}

	// error checking
	{
		if len(a.Outs.Dims) == 0 {
			return errors.Errorf("Can't Init() pool node, len(Dims) == 0")
		}
		if len(a.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() pool node, len(InputDims) == 0")
		}
		if len(a.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() pool node, len(InputDims) != len(Dims) (len(%v) != len(%v))", a.Ins.Dims, a.Outs.Dims)
		}
		if len(a.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() pool node, len(Filter) != len(Dims) (len(%v) != len(%v))", a.Filter.Dims, a.Outs.Dims)
		}
		if len(a.Stride) != numDims {
			return errors.Errorf("Can't Init() pool node, len(Stride) != len(Dims) (len(%v) != len(%v))", a.Stride, a.Outs.Dims)
		}

		// check for bad values
		for i, d := range a.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}

		// check that the size of the node is equal to the product of dimensions
		size := 1
		for _, d := range a.Outs.Dims {
			size *= d
		}
		if size != n.Size() {
			return errors.Errorf("Can't Init() pool node, volume of dimensions is not equal to node size (%d != %d)", size, n.Size())
		}

		// check that the size fo the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range a.Ins.Dims {
			inputSize *= d
		}
		if inputSize != n.NumInputs() {
			return errors.Errorf("Can't Init() pool node, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, n.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size) / (stride) + 1
		for i := range a.Outs.Dims {
			numerator := a.Ins.Dims[i] - a.Filter.Dims[i]

			if numerator%a.Stride[i] != 0 {
				return errors.Errorf("Can't Init() pool node, stride does not fit Dims[%d] and filter ((%d - %d) %% %d != 0)", i, a.Ins.Dims[i], a.Filter.Dims[i], a.Stride[i])
			} else if a.Outs.Dims[i] != numerator/a.Stride[i]+1 {
				return errors.Errorf("Can't Init() pool node, expected Dims[%d] does not match given (%d != %d)", i, numerator/a.Stride[i]+1, a.Outs.Dims[i])
			}
		}
	}

	return nil
}

func (a *avgPool) Save(n *bs.Node, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Couldn't save operator: failed to create directory to house save file")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't save operator: failed to create file 'weights.txt'")
	}

	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(a); err != nil {
		return errors.Wrapf(err, "Couldn't save operator: failed to encode JSON to file\n")
	}

	return nil
}

func (a *avgPool) Load(n *bs.Node, dirPath string, aux []interface{}) error {

	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't load operator: could not open file 'weights.txt'")
	}

	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(a); err != nil {
		return errors.Wrapf(err, "Couldn't load operator: failed to decode JSON from file\n")
	}

	return nil
}

func (a *avgPool) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	filterSize := 1
	for _, d := range a.Filter.Dims {
		filterSize *= d
	}

	calculateValue := func(index int) {
		out := a.Outs.Point(index)

		var avg float64
		f := make([]int, len(a.Filter.Dims))

		for i := 0; i < filterSize; i++ {

			in := make([]int, len(a.Outs.Dims))
			for i := range in {
				in[i] = a.Stride[i]*out[i] + f[i]
			}

			avg += inputs[a.Ins.Index(in)]
		}

		values[index] = avg / float64(filterSize)
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

func (a *avgPool) Value(n *bs.Node, index int) float64 {
	panic("Cannot get value for pool layer")
}

func (a *avgPool) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {

	filterSize := 1
	for _, d := range a.Filter.Dims {
		filterSize *= d
	}

	sendDelta := func(inputIndex int) {
		input := a.Ins.Point(inputIndex)

		// largets values of 'out'
		out_init := make([]int, len(a.Outs.Dims))

		// smallest values of 'f'
		f_init := make([]int, len(a.Filter.Dims))

		for i := range out_init {
			l := n.NumInputs()
			n := input[i]
			f := a.Filter.Dims[i]
			s := a.Stride[i]

			r := n + f - l
			o := n / s

			if s*o+f >= l {
				if r%s != 0 {
					out_init[i] = o - (r / s) - 1
				} else {
					out_init[i] = o - (r / s)
				}
			} else {
				out_init[i] = o
			}

			f_init[i] = n - (s * out_init[i])

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
			if f[len(f)-1] >= a.Filter.Dims[len(f)-1] || out[len(out)-1] < 0 {
				break
			}

			// add delta to sum
			if a.Outs.Index(out) >= n.Size() || a.Outs.Index(out) < 0 {
				fmt.Println(out, out_init, a.Outs.Index(out))
			}

			sum += n.Delta(a.Outs.Index(out))

			// increment:
			// for each value in f
			for i := range f {
				// increment f[i] (decrement out[i]) by stride[i]
				f[i] += a.Stride[i]
				out[i] -= a.Stride[i]

				if f[i] < a.Filter.Dims[i] && out[i] >= 0 {
					break
				}

				if i == len(f)-1 {
					break
				}

				f[i] = f_init[i]
				out[i] = out_init[i]
			}
		}

		add(inputIndex, sum/float64(filterSize))
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(start, end, sendDelta, opsPerThread, threadsPerCPU)

	return nil
}

func (a *avgPool) CanBeAdjusted(n *bs.Node) bool {
	return false
}

// for now, we're saying that it needs values because it's too hard to rework it right now
// Yay! Technical debt!
func (a *avgPool) NeedsValues(n *bs.Node) bool {
	return true
}

func (a *avgPool) NeedsInputs(n *bs.Node) bool {
	return true
}

func (a *avgPool) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (a *avgPool) AddWeights(n *bs.Node) error {
	return nil
}

// ***************************************************
// MaxPool:
// ***************************************************

func (mp *maxPool) Init(n *bs.Node) error {

	numDims := len(mp.Outs.Dims)

	// set defaults
	if mp.Stride == nil {
		mp.Stride = make([]int, numDims)
		for i := range mp.Stride {
			mp.Stride[i] = 1
		}
	}

	// error checking
	{
		if len(mp.Outs.Dims) == 0 {
			return errors.Errorf("Can't Init() pool node, len(Dims) == 0")
		}
		if len(mp.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() pool node, len(InputDims) == 0")
		}
		if len(mp.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() pool node, len(InputDims) != len(Dims) (len(%v) != len(%v))", mp.Ins.Dims, mp.Outs.Dims)
		}
		if len(mp.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() pool node, len(Filter) != len(Dims) (len(%v) != len(%v))", mp.Filter.Dims, mp.Outs.Dims)
		}
		if len(mp.Stride) != numDims {
			return errors.Errorf("Can't Init() pool node, len(Stride) != len(Dims) (len(%v) != len(%v))", mp.Stride, mp.Outs.Dims)
		}

		// check for bad values
		for i, d := range mp.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() pool node, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}

		// check that the size of the node is equal to the product of dimensions
		size := 1
		for _, d := range mp.Outs.Dims {
			size *= d
		}
		if size != n.Size() {
			return errors.Errorf("Can't Init() pool node, volume of dimensions is not equal to node size (%d != %d)", size, n.Size())
		}

		// check that the size fo the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range mp.Ins.Dims {
			inputSize *= d
		}
		if inputSize != n.NumInputs() {
			return errors.Errorf("Can't Init() pool node, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, n.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size) / (stride) + 1
		for i := range mp.Outs.Dims {
			numerator := mp.Ins.Dims[i] - mp.Filter.Dims[i]

			if numerator%mp.Stride[i] != 0 {
				return errors.Errorf("Can't Init() pool node, stride does not fit Dims[%d] and filter ((%d - %d) %% %d != 0)", i, mp.Ins.Dims[i], mp.Filter.Dims[i], mp.Stride[i])
			} else if mp.Outs.Dims[i] != numerator/mp.Stride[i]+1 {
				return errors.Errorf("Can't Init() pool node, expected Dims[%d] does not match given (%d != %d)", i, numerator/mp.Stride[i]+1, mp.Outs.Dims[i])
			}
		}
	}

	return nil
}

func (mp *maxPool) Save(n *bs.Node, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Couldn't save operator: failed to create directory to house save file")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't save operator: failed to create file 'weights.txt'")
	}

	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(mp); err != nil {
		return errors.Wrapf(err, "Couldn't save operator: failed to encode JSON to file\n")
	}

	return nil
}

func (mp *maxPool) Load(n *bs.Node, dirPath string, aux []interface{}) error {

	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't load operator: could not open file 'weights.txt'")
	}

	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(mp); err != nil {
		return errors.Wrapf(err, "Couldn't load operator: failed to decode JSON from file\n")
	}

	return nil
}

func (mp *maxPool) Evaluate(n *bs.Node, values []float64) error {
	inputs := n.CopyOfInputs()

	filterSize := 1
	for _, d := range mp.Filter.Dims {
		filterSize *= d
	}

	mp.switches = make([]int, len(values))

	calculateValue := func(index int) {
		out := mp.Outs.Point(index)

		var max float64
		f := make([]int, len(mp.Filter.Dims))

		for i := 0; i < filterSize; i++ {

			in := make([]int, len(mp.Outs.Dims))
			for i := range in {
				in[i] = mp.Stride[i]*out[i] + f[i]
			}

			if i == 0 {
				mp.switches[index] = mp.Ins.Index(in)
				max = inputs[mp.switches[index]]
			} else {
				s := mp.Ins.Index(in)
				if inputs[s] > max {
					max = inputs[s]
					mp.switches[index] = s
				}

			}
		}

		values[index] = max
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

func (mp *maxPool) Value(n *bs.Node, index int) float64 {
	panic("Cannot get value for pool layer")
}

func (mp *maxPool) InputDeltas(n *bs.Node, add func(int, float64), start, end int) error {

	sendDelta := func(inputIndex int) {
		input := mp.Ins.Point(inputIndex)

		// largets values of 'out'
		out_init := make([]int, len(mp.Outs.Dims))

		// smallest values of 'f'
		f_init := make([]int, len(mp.Filter.Dims))

		for i := range out_init {
			l := n.NumInputs()
			n := input[i]
			f := mp.Filter.Dims[i]
			s := mp.Stride[i]

			r := n + f - l
			o := n / s

			if s*o+f >= l {
				if r%s != 0 {
					out_init[i] = o - (r / s) - 1
				} else {
					out_init[i] = o - (r / s)
				}
			} else {
				out_init[i] = o
			}

			f_init[i] = n - (s * out_init[i])

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
			if f[len(f)-1] >= mp.Filter.Dims[len(f)-1] || out[len(out)-1] < 0 {
				break
			}

			// add delta if it's used
			outIndex := mp.Outs.Index(out)
			if mp.switches[outIndex] == inputIndex {
				sum += n.Delta(outIndex)
			}

			// increment:
			// for each value in f
			for i := range f {
				// increment f[i] (decrement out[i]) by stride[i]
				f[i] += mp.Stride[i]
				out[i] -= mp.Stride[i]

				if f[i] < mp.Filter.Dims[i] && out[i] >= 0 {
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

func (mp *maxPool) CanBeAdjusted(n *bs.Node) bool {
	return false
}

// for now, we're saying that it needs values because it's too hard to rework it right now
// Yay! Technical debt!
func (mp *maxPool) NeedsValues(n *bs.Node) bool {
	return true
}

func (mp *maxPool) NeedsInputs(n *bs.Node) bool {
	return true
}

func (mp *maxPool) Adjust(n *bs.Node, learningRate float64, saveChanges bool) error {
	return nil
}

func (mp *maxPool) AddWeights(n *bs.Node) error {
	return nil
}
