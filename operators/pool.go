package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"

	"encoding/json"
	"os"
	"fmt"
)

type PoolArgs struct {
	// The dimensions of the output -- usually will be {width, height, depth...}
	// Init() will return error if the product of 'Dims' multiplied together doesn't
	// equal the number of values in the layer - *Layer.Size()
	//
	// should be the same length as 'InputDims'
	//
	// while this could have been designed to not need 'Dims', it is useful for double-checking
	Dims []int

	// The dimensions of the input
	// Init() checks that the product of 'InputDims' multiplied together equals
	// the total number of input values to the Layer
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

type leakyMaxPool struct {
	maxPool

	// should be in range [0, 1]
	LeakFactor float64
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

// can be supplied 'nil' if being used to load from file
func LeakyMaxPool(args *PoolArgs, leakFactor float64) *leakyMaxPool {
	return &leakyMaxPool{maxPool{constructPool(args), nil}, leakFactor}
}

// ***************************************************
// AvgPool:
// ***************************************************

func (a *avgPool) Init(l *badstudent.Layer) error {

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
			return errors.Errorf("Can't Init() pool layer, len(Dims) == 0")
		}
		if len(a.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) == 0")
		}
		if len(a.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) != len(Dims) (len(%v) != len(%v))", a.Ins.Dims, a.Outs.Dims)
		}
		if len(a.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Filter) != len(Dims) (len(%v) != len(%v))", a.Filter.Dims, a.Outs.Dims)
		}
		if len(a.Stride) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Stride) != len(Dims) (len(%v) != len(%v))", a.Stride, a.Outs.Dims)
		}

		// check for bad values
		for i, d := range a.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range a.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}

		// check that the size of the layer is equal to the product of dimensions
		size := 1
		for _, d := range a.Outs.Dims {
			size *= d
		}
		if size != l.Size() {
			return errors.Errorf("Can't Init() pool layer, volume of dimensions is not equal to layer size (%d != %d)", size, l.Size())
		}

		// check that the size fo the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range a.Ins.Dims {
			inputSize *= d
		}
		if inputSize != l.NumInputs() {
			return errors.Errorf("Can't Init() pool layer, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, l.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size) / (stride) + 1
		for i := range a.Outs.Dims {
			numerator := a.Ins.Dims[i] - a.Filter.Dims[i]

			if numerator%a.Stride[i] != 0 {
				return errors.Errorf("Can't Init() pool layer, stride does not fit Dims[%d] and filter ((%d - %d) %% %d != 0)", i, a.Ins.Dims[i], a.Filter.Dims[i], a.Stride[i])
			} else if a.Outs.Dims[i] != numerator/a.Stride[i]+1 {
				return errors.Errorf("Can't Init() pool layer, expected Dims[%d] does not match given (%d != %d)", i, numerator/a.Stride[i]+1, a.Outs.Dims[i])
			}
		}
	}

	return nil
}

func (a *avgPool) Save(l *badstudent.Layer, dirPath string) error {
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

func (a *avgPool) Load(l *badstudent.Layer, dirPath string, aux []interface{}) error {

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

func (a *avgPool) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()

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

func (a *avgPool) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

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
			n := input[i]
			f := a.Filter.Dims[i]
			l := l.NumInputs()
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
			if a.Outs.Index(out) >= l.Size() || a.Outs.Index(out) < 0 {
				fmt.Println(out, out_init, a.Outs.Index(out))
			}

			sum += l.Delta(a.Outs.Index(out))

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

func (a *avgPool) CanBeAdjusted(l *badstudent.Layer) bool {
	return false
}

func (a *avgPool) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	return nil
}

func (a *avgPool) AddWeights(l *badstudent.Layer) error {
	return nil
}

// ***************************************************
// MaxPool:
// ***************************************************

func (mp *maxPool) Init(l *badstudent.Layer) error {

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
			return errors.Errorf("Can't Init() pool layer, len(Dims) == 0")
		}
		if len(mp.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) == 0")
		}
		if len(mp.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) != len(Dims) (len(%v) != len(%v))", mp.Ins.Dims, mp.Outs.Dims)
		}
		if len(mp.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Filter) != len(Dims) (len(%v) != len(%v))", mp.Filter.Dims, mp.Outs.Dims)
		}
		if len(mp.Stride) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Stride) != len(Dims) (len(%v) != len(%v))", mp.Stride, mp.Outs.Dims)
		}

		// check for bad values
		for i, d := range mp.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range mp.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}

		// check that the size of the layer is equal to the product of dimensions
		size := 1
		for _, d := range mp.Outs.Dims {
			size *= d
		}
		if size != l.Size() {
			return errors.Errorf("Can't Init() pool layer, volume of dimensions is not equal to layer size (%d != %d)", size, l.Size())
		}

		// check that the size fo the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range mp.Ins.Dims {
			inputSize *= d
		}
		if inputSize != l.NumInputs() {
			return errors.Errorf("Can't Init() pool layer, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, l.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size) / (stride) + 1
		for i := range mp.Outs.Dims {
			numerator := mp.Ins.Dims[i] - mp.Filter.Dims[i]

			if numerator%mp.Stride[i] != 0 {
				return errors.Errorf("Can't Init() pool layer, stride does not fit Dims[%d] and filter ((%d - %d) %% %d != 0)", i, mp.Ins.Dims[i], mp.Filter.Dims[i], mp.Stride[i])
			} else if mp.Outs.Dims[i] != numerator/mp.Stride[i]+1 {
				return errors.Errorf("Can't Init() pool layer, expected Dims[%d] does not match given (%d != %d)", i, numerator/mp.Stride[i]+1, mp.Outs.Dims[i])
			}
		}
	}

	return nil
}

func (mp *maxPool) Save(l *badstudent.Layer, dirPath string) error {
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

func (mp *maxPool) Load(l *badstudent.Layer, dirPath string, aux []interface{}) error {

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

func (mp *maxPool) Evaluate(l *badstudent.Layer, values []float64) error {
	inputs := l.CopyOfInputs()

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

func (mp *maxPool) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

	sendDelta := func(inputIndex int) {
		input := mp.Ins.Point(inputIndex)

		// largets values of 'out'
		out_init := make([]int, len(mp.Outs.Dims))

		// smallest values of 'f'
		f_init := make([]int, len(mp.Filter.Dims))

		for i := range out_init {
			n := input[i]
			f := mp.Filter.Dims[i]
			l := l.NumInputs()
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
				sum += l.Delta(outIndex)
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

func (mp *maxPool) CanBeAdjusted(l *badstudent.Layer) bool {
	return false
}

func (mp *maxPool) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	return nil
}

func (mp *maxPool) AddWeights(l *badstudent.Layer) error {
	return nil
}

// ***************************************************
// LeakyMaxPool:
// ***************************************************

func (lmp *leakyMaxPool) Init(l *badstudent.Layer) error {

	numDims := len(lmp.Outs.Dims)

	// set defaults
	if lmp.Stride == nil {
		lmp.Stride = make([]int, numDims)
		for i := range lmp.Stride {
			lmp.Stride[i] = 1
		}
	}

	// error checking
	{
		if len(lmp.Outs.Dims) == 0 {
			return errors.Errorf("Can't Init() pool layer, len(Dims) == 0")
		}
		if len(lmp.Ins.Dims) == 0 {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) == 0")
		}
		if len(lmp.Ins.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(InputDims) != len(Dims) (len(%v) != len(%v))", lmp.Ins.Dims, lmp.Outs.Dims)
		}
		if len(lmp.Filter.Dims) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Filter) != len(Dims) (len(%v) != len(%v))", lmp.Filter.Dims, lmp.Outs.Dims)
		}
		if len(lmp.Stride) != numDims {
			return errors.Errorf("Can't Init() pool layer, len(Stride) != len(Dims) (len(%v) != len(%v))", lmp.Stride, lmp.Outs.Dims)
		}

		// check for bad values
		for i, d := range lmp.Outs.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Dims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range lmp.Ins.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, InputDims[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range lmp.Filter.Dims {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Filter[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}
		for i, d := range lmp.Stride {
			if d < 1 {
				return errors.Errorf("Can't Init() pool layer, Stride[%d] = %d. All dimension values should be ≥ 1", i, d)
			}
		}

		// check that the size of the layer is equal to the product of dimensions
		size := 1
		for _, d := range lmp.Outs.Dims {
			size *= d
		}
		if size != l.Size() {
			return errors.Errorf("Can't Init() pool layer, volume of dimensions is not equal to layer size (%d != %d)", size, l.Size())
		}

		// check that the size fo the inputs is equal to the product of input dimensions
		inputSize := 1
		for _, d := range lmp.Ins.Dims {
			inputSize *= d
		}
		if inputSize != l.NumInputs() {
			return errors.Errorf("Can't Init() pool layer, product of input dimensions is not equal to number of inputs (%d != %d)", inputSize, l.NumInputs())
		}

		// check that the dimensions of the output volume add up
		// the length of the side of the output volume can be determined by:
		// (input length - filter size) / (stride) + 1
		for i := range lmp.Outs.Dims {
			numerator := lmp.Ins.Dims[i] - lmp.Filter.Dims[i]

			if numerator%lmp.Stride[i] != 0 {
				return errors.Errorf("Can't Init() pool layer, stride does not fit Dims[%d] and filter ((%d - %d) %% %d != 0)", i, lmp.Ins.Dims[i], lmp.Filter.Dims[i], lmp.Stride[i])
			} else if lmp.Outs.Dims[i] != numerator/lmp.Stride[i]+1 {
				return errors.Errorf("Can't Init() pool layer, expected Dims[%d] does not match given (%d != %d)", i, numerator/lmp.Stride[i]+1, lmp.Outs.Dims[i])
			}
		}

		// leakyMaxPool requires a filter size > 1
		filterSize := 1
		for _, d := range lmp.Filter.Dims {
			filterSize *= d
		}
		if filterSize == 1 { // we know that everything is >= 1
			return errors.Errorf("Can't Init() pool layer, filter size must be > 1 (product of: %v)", lmp.Filter.Dims)
		}
	}

	return nil
}

func (lmp *leakyMaxPool) Save(l *badstudent.Layer, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Couldn't save operator: failed to create directory to house save file")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't save operator: failed to create file 'weights.txt'")
	}

	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(lmp); err != nil {
		return errors.Wrapf(err, "Couldn't save operator: failed to encode JSON to file\n")
	}

	return nil
}

func (lmp *leakyMaxPool) Load(l *badstudent.Layer, dirPath string, aux []interface{}) error {

	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't load operator: could not open file 'weights.txt'")
	}

	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(lmp); err != nil {
		return errors.Wrapf(err, "Couldn't load operator: failed to decode JSON from file\n")
	}

	return nil
}

func (lmp *leakyMaxPool) Evaluate(l *badstudent.Layer, values []float64) error {

	inputs := l.CopyOfInputs()

	filterSize := 1
	for _, d := range lmp.Filter.Dims {
		filterSize *= d
	}

	lmp.switches = make([]int, len(values))

	manyLeak := (1 - lmp.LeakFactor) / float64(filterSize-1)

	calculateValue := func(index int) {
		out := lmp.Outs.Point(index)

		var sum float64

		f := make([]int, len(lmp.Filter.Dims))

		for i := 0; i < filterSize; i++ {

			in := make([]int, len(lmp.Outs.Dims))
			for i := range in {
				in[i] = lmp.Stride[i]*out[i] + f[i]
			}

			if i == 0 {
				lmp.switches[index] = lmp.Ins.Index(in)
				sum += lmp.LeakFactor * inputs[lmp.switches[index]]
			} else {
				s := lmp.Ins.Index(in)
				if inputs[s] > inputs[lmp.switches[index]] {
					sum += (lmp.LeakFactor * inputs[s]) - (manyLeak * inputs[lmp.switches[index]])
					lmp.switches[index] = s
				}
			}
		}

		values[index] = sum
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

func (lmp *leakyMaxPool) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {
	filterSize := 1
	for _, d := range lmp.Filter.Dims {
		filterSize *= d
	}

	manyLeak := (1 - lmp.LeakFactor) / float64(filterSize-1)

	sendDelta := func(inputIndex int) {
		input := lmp.Ins.Point(inputIndex)

		// largets values of 'out'
		out_init := make([]int, len(lmp.Outs.Dims))

		// smallest values of 'f'
		f_init := make([]int, len(lmp.Filter.Dims))

		for i := range out_init {
			n := input[i]
			f := lmp.Filter.Dims[i]
			l := l.NumInputs()
			s := lmp.Stride[i]

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
			if f[len(f)-1] >= lmp.Filter.Dims[len(f)-1] || out[len(out)-1] < 0 {
				break
			}

			// add delta to sum
			outIndex := lmp.Outs.Index(out)
			if lmp.switches[outIndex] == inputIndex {
				sum += l.Delta(outIndex) * lmp.LeakFactor
			} else {
				sum += l.Delta(outIndex) * manyLeak
			}

			// increment:
			// for each value in f
			for i := range f {
				// increment f[i] (decrement out[i]) by stride[i]
				f[i] += lmp.Stride[i]
				out[i] -= lmp.Stride[i]

				if f[i] < lmp.Filter.Dims[i] && out[i] >= 0 {
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

func (lmp *leakyMaxPool) CanBeAdjusted(l *badstudent.Layer) bool {
	return false
}

func (lmp *leakyMaxPool) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	return nil
}

func (lmp *leakyMaxPool) AddWeights(l *badstudent.Layer) error {
	return nil
}
