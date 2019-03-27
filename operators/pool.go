package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"github.com/sharnoff/tensors"
)

type poolConstructor struct {
	dims      []int
	inputDims []int
	filter    []int
}

type pool struct {
	*poolConstructor

	Outs *utils.MultiDim
	Ins  *utils.MultiDim
	// Filter
	Filt *utils.MultiDim

	// Stride
	Str     []int
	Padding []int

	PaddingValue float64
}

func basePool() pool {
	p := pool{poolConstructor: new(poolConstructor)}
	p.PaddingValue = defaultValue["pool-padding"]
	return p
}

// ***************************************************
// Customization Functions
//
// Note: All functions are repated for both pooling operators because golang does not support
// inheritance.
// ***************************************************

// Dims sets the output dimensions of the pooling Operator. It does not check that the dimensions
// are valid until it is Finalized. Dims will panic if called after the pooling Operator has been
// Finalized.
//
// Dims is optional, as it can be calculated from other information, but it can also be provided.
//
// If any dimensions collapse to a size of 1, they must still be included.
func (p *avgPool) Dims(dims ...int) *avgPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.dims = dims
	return p
}

// see above.
func (p *maxPool) Dims(dims ...int) *maxPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.dims = dims
	return p
}

// InputDims sets the input dimensions of the pooling Operator, for internal use.
// This is REQUIRED unless the Operator is being loaded. InputDims will panic if
// called after the pooling Operator has been Finalized.
func (p *avgPool) InputDims(dims ...int) *avgPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.inputDims = dims
	return p
}

// see above.
func (p *maxPool) InputDims(dims ...int) *maxPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.inputDims = dims
	return p
}

// Filter sets the size of the pooling for each dimension. This is REQUIRED unless
// the Operator is being loaded. Filter will panic if called after the pooling
// Operator has been Finalized.
func (p *avgPool) Filter(dims ...int) *avgPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.filter = dims
	return p
}

// see above.
func (p *maxPool) Filter(dims ...int) *maxPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.filter = dims
	return p
}

// Stride sets the space between centers of pooling regions. Stride defaults to the
// same size as filter (pooling region). Stride will panic if called after the
// pooling Operator has been Finalized.
//
// At finalization, pooling Operators will return error if any dimensions of Stride
// are larger than Filter.
//
// Stride is optional.
func (p *avgPool) Stride(dims ...int) *avgPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.Str = dims
	return p
}

// see above.
func (p *maxPool) Stride(dims ...int) *maxPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.Str = dims
	return p
}

// Pad sets the amount of padding on both ends of each dimension. Pad defaults to
// none, if not provided. Pad will panic if called after the pooling Operator has
// been Finalized.
func (p *avgPool) Pad(dims ...int) *avgPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.Padding = dims
	return p
}

// see above.
func (p *maxPool) Pad(dims ...int) *maxPool {
	if p.poolConstructor == nil {
		panic("pooling Operator has been finalized")
	}

	p.Padding = dims
	return p
}

// PadValue sets the value of the padding around the input. The default can be set
// by SetDefault("pool-padding").
func (p *avgPool) PadValue(v float64) *avgPool {
	p.PaddingValue = v
	return p
}

func (p *maxPool) PadValue(v float64) *maxPool {
	p.PaddingValue = v
	return p
}

type avgPool struct {
	pool
}

type maxPool struct {
	pool

	// the index (in inputs) of the highest value
	switches []int
}

// AvgPool returns the average pooling function, which implements
// badstudent.Operator. AvgPool can be customized with the methods available on
// pool. Setting InputDims and FilterSize is required.
//
// The default value of padding can be set by SetDefault("pool-padding").
//
// For more information, see: http://cs231n.github.io/convolutional-networks/#pool
func AvgPool() *avgPool {
	ap := basePool()
	return &avgPool{ap}
}

// MaxPool returns the max-pooling function, which implements badstudent.Operator.
// MaxPool can be customized with the methods available on pool. Setting InputDims
// and FilterSize is required.
//
// N.B.: MaxPool cannot be used in a node with Delay, because the functioning of
// saving switches does not work with calculating deltas.
//
// The default value of padding can be set by SetDefault("pool-padding").
//
// For more information, see: http://cs231n.github.io/convolutional-networks/#pool
func MaxPool() *maxPool {
	mp := &maxPool{basePool(), nil}
	return mp
}

// ***************************************************
// Basic list operations:
// ***************************************************

func mapAdd(base, addend []int) []int {
	for i := range base {
		base[i] += addend[i]
	}
	return base
}

func mapMult(base, factor []int) []int {
	for i := range base {
		base[i] *= factor[i]
	}
	return base
}

func mapSub(base, subtrahend []int) []int {
	for i := range base {
		base[i] -= subtrahend[i]
	}
	return base
}

// ***************************************************
// Shared Methods / Helper functions:
// ***************************************************

// MustSize calls Size, but panics if it encounters an error.
func (p *pool) MustSize() int {
	size, err := p.Size()
	if err != nil {
		panic(err.Error())
	}

	return size
}

// Size returns the expected size of the pooling Operator, before it has been
// finalized. If the configuration is invalid, it will return error, just as
// Finalize would.
func (p *pool) Size() (int, error) {
	if p.Outs != nil {
		return p.Outs.Size(), nil
	}

	if p.inputDims == nil {
		return 0, errors.Errorf("InputDims has not been set")
	} else if p.filter == nil {
		return 0, errors.Errorf("Filter has not been set")
	}

	// check that none of them have length zero or values <= 1
	dimList := [][]int{p.inputDims, p.dims, p.filter, p.Str, p.Padding}
	names := []string{"InputDims", "Dims", "Filter", "Stride", "Padding"} // only used in case of error
	for i := range dimList {
		if dimList[i] == nil {
			continue
		}

		if len(dimList[i]) == 0 {
			return 0, errors.Errorf("%s has been set with length zero", names[i])
		} else if len(dimList[i]) != len(p.inputDims) {
			return 0, errors.Errorf("%s has different length to InputDims (%d != %d)", names[i], len(dimList[i]), len(p.inputDims))
		}

		for d := range dimList[i] {
			if dimList[i][d] < 1 {
				return 0, errors.Errorf("%s[%d] = %d, must be ≥ 1", names[i], d, dimList[i][d])
			}
		}
	}

	if p.Str == nil {
		p.Str = p.filter
	} else {
		for d := range p.Str {
			if p.Str[d] > p.filter[d] {
				return 0, errors.Errorf("Filter[%d] is less than than Stride[%d] (%d > %d)", d, d, p.filter[d], p.Str[d])
			}
		}
	}

	if p.Padding == nil {
		p.Padding = make([]int, len(p.inputDims))
	}

	if p.dims != nil { // if everything is filled in, check whether or not it works
		for i := range p.inputDims {
			in := p.inputDims[i] + 2*p.Padding[i]
			d := p.dims[i]
			f := p.filter[i]
			s := p.Str[i]

			if (in+s-f)%s != 0 {
				return 0, errors.Errorf("Dimenision #%d does not divide evenly: (InputDim + 2*Padding + Stride - Filter) %% (Stride) != 0 "+
					"((%d + 2*%d + %d - %d)%%%d != 0)", i, p.inputDims[i], p.Padding[i], s, f, s)
			} else if (in+s-f)/s != d {
				return 0, errors.Errorf("Dimension #%d does not produce desired output (InputDim + 2*Padding + Stride - Filter) / (Stride) != OutputDim "+
					"((%d + 2*%d + %d - %d) / (%d) != %d", i, p.inputDims[i], p.Padding[i], s, f, s, d)
			}
		}
	} else { // fill in
		p.dims = make([]int, len(p.inputDims))

		for i := range p.inputDims {
			in := p.inputDims[i] + 2*p.Padding[i]
			f := p.filter[i]
			s := p.Str[i]

			if (in+s-f)%s != 0 {
				return 0, errors.Errorf("Dimenision #%d does not divide evenly: (InputDim + 2*Padding + Stride - Filter) %% (Stride) != 0 "+
					"((%d + 2*%d + %d - %d)%%%d != 0)", i, p.inputDims[i], p.Padding[i], s, f, s)
			}

			p.dims[i] = (in + s - f) / s
		}
	}

	p.Outs = utils.NewMultiDim(p.dims)
	p.Ins = utils.NewMultiDim(p.inputDims)
	p.Filt = utils.NewMultiDim(p.filter)

	p.poolConstructor = nil

	return p.Outs.Size(), nil
}

func (p *pool) Finalize(n *bs.Node) error {
	_, err := p.Size()
	if err != nil {
		return err
	}

	if p.Ins.Size() != n.NumInputs() {
		return errors.Errorf("Mismatch between expected number of inputs and actual (%d != %d)", p.Ins.Size(), n.NumInputs())
	}

	return nil
}

func (p *pool) OutputShape(inputs []*bs.Node) (tensors.Tensor, error) {
	_, err := p.Size()
	if err != nil {
		return tensors.Tensor{}, err
	}

	return tensors.NewTensor(p.Outs.Dims), nil
}

func (p *pool) Get() interface{} {
	return *p
}

func (p *pool) Blank() interface{} {
	return p
}

func (po *pool) isPadding(point []int) bool {
	p := make([]int, len(point))
	copy(p, point)
	mapSub(p, po.Padding)

	// if any are less than 0, it's padding
	for i := range p {
		if p[i] < 0 || p[i] > po.Outs.Dim(i) {
			return true
		}
	}

	return false
}

func (p *pool) inputsTo(index int) []int {
	// here, underscores are used as a suffix to indicate the type of the
	// variable. For example, x_i would be an index with name 'x', and x_p would
	// be an n-dimensional point with name 'x'.

	// topleft is the top-left base point for the filter, including padding
	topLeft_p := mapMult(p.Outs.Point(index), p.Str)

	// the list of input indexes to be supplied to c
	inList := make([]int, p.Filt.Size())

	// the point in relation to the top left of the filter
	fMod := make([]int, len(p.Str)) // stride is the easiest way to get the number of dimensions
	for i := 0; i < p.Filt.Size(); i++ {

		// in is the current input that we're looking at, including padding
		var in_p []int
		{
			in_p = make([]int, len(topLeft_p))
			copy(in_p, topLeft_p)
			mapAdd(in_p, fMod)
		}

		if p.isPadding(in_p) {
			// -1 indicates that it is out of range of the inputs -- it's padding
			inList[i] = -1
		} else {
			inList[i] = p.Ins.Index(mapSub(in_p, p.Padding))
		}

		if i+1 < p.Filt.Size() {
			p.Filt.Increment(fMod)
		}
	}

	return inList
}

// ***************************************************
// AvgPool:
// ***************************************************

func (t *avgPool) TypeString() string {
	return "avg-pool"
}

func (t *avgPool) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()

	f := func(v int) {
		ins := t.inputsTo(v)
		var sum float64
		for _, in := range ins {
			if in == -1 {
				sum += t.PaddingValue
			} else {
				sum += inputs[in]
			}
		}
		values[v] = sum / float64(len(ins))
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t *avgPool) InputDeltas(n *bs.Node) []float64 {
	atoms := make([]uint64, n.NumInputs())

	f := func(out int) {
		ins := t.inputsTo(out)

		for _, in := range ins {
			if in != -1 {
				atomAdd(&atoms[in], n.Delta(out)/float64(len(ins)))
			}
		}
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)

	return uint64ToFloat64(atoms)
}

// ***************************************************
// MaxPool:
// ***************************************************

func (t *maxPool) TypeString() string {
	return "max-pool"
}

func (t *maxPool) Finalize(n *bs.Node) error {
	if err := t.pool.Finalize(n); err != nil {
		return err
	} else if n.Delay() != 0 {
		return errors.Errorf("MaxPooling cannot have delay (n.Delay() = %d)", n.Delay())
	}

	return nil
}

func (t *maxPool) Evaluate(n *bs.Node, values []float64) {
	inputs := n.AllInputs()

	// note: switches will sometimes be zero
	t.switches = make([]int, len(values))

	f := func(v int) {
		ins := t.inputsTo(v)

		max := t.PaddingValue
		if ins[0] != -1 { // if it's not padding
			max = inputs[ins[0]]
		}
		t.switches[v] = ins[0]

		for i := 1; i < len(ins); i++ {
			v := t.PaddingValue
			if ins[i] != -1 {
				v = inputs[ins[i]]
			}

			if v > max {
				max, t.switches[i] = v, ins[i]
			}
		}

		values[v] = max
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), f, opsPerThread, threadsPerCPU)
}

func (t *maxPool) InputDeltas(n *bs.Node) []float64 {
	atoms := make([]uint64, n.NumInputs())

	f := func(out int) {
		if t.switches[out] != -1 {
			atomAdd(&atoms[t.switches[out]], n.Delta(out))
		}
	}

	opsPerThread, threadsPerCPU := 10, 1
	utils.MultiThread(0, n.Size(), f, opsPerThread, threadsPerCPU)

	return uint64ToFloat64(atoms)
}
