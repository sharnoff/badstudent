package utils

// Allows storage of n-dimensional slices
//
// stored such that the oscillation frequency of the dimensions decreases as
// the index in dimensions increases
//
// made public for use in mult-dimensional operators
//
// the fields are made public in order to allow exporting to JSON,
// but they should not actually be altered once it has been initialized
type MultiDim struct {
	// the width, height, depth, etc. of each dimension
	Dims []int

	// the number of values encapsulated by a 'set' of this dimension
	// -- sizes[0] = dims[0]; sizes[end] = len(base)
	// sizes will be initialized by the constructor -- should not be provided
	Sizes []int
}

// Creates a new MultiDim slice wrapper
//
// assumes that the product of 'dims' multiplied together is equal to the length of 'base'
// stored as: z{y{x, x}, y{x, x}}, z{y{x, x}, y{x, x}}.
// accessed as: [x, y, z]
func NewMultiDim(dims []int) *MultiDim {
	m := &MultiDim{
		Dims:  dims,
		Sizes: make([]int, len(dims)),
	}

	m.Sizes[0] = m.Dims[0]

	for i := 1; i < len(m.Sizes); i++ {
		m.Sizes[i] = m.Sizes[i-1] * m.Dims[i]
	}

	return m
}

// returns the index corresponding to the given point
// assumes that the point has the same number of dimensions as 'm'
//
// the given point should have the dimensions in the same order as they were originally given
func (m *MultiDim) Index(point []int) int {
	index := point[0]
	for i := 1; i < len(m.Sizes); i++ {
		index += point[i] * m.Sizes[i-1]
	}

	return index
}

// returns the multi-dimensional point leading to the given index in the base array
//
// assumes that the given index will be in bounds
func (m *MultiDim) Point(index int) []int {
	p := make([]int, len(m.Dims))
	for i := len(p) - 1; i >= 1; i-- { // doesn't go to 0
		p[i] = index / m.Sizes[i-1]
		index = index % m.Sizes[i-1]
	}

	p[0] = index
	return p
}

func (m *MultiDim) Size() int {
	return m.Sizes[len(m.Sizes)-1]
}

func (m *MultiDim) Dim(d int) int {
	return m.Dims[d]
}

// Increments the given point by 1
// assumes that len(point) = len(dims)
//
// if it overflows, it leaves it at the highest possible value
// returns false if it overflows, else returns true
func (m *MultiDim) Increment(point []int) bool {
	for i := range point {
		point[i]++
		if point[i] < m.Dims[i] {
			break
		}

		if i == len(point)-1 {
			return false
		}

		point[i] = 0
	}

	return true
}

// increments the point by the amount specified
// cannot accept negative numbers
//
// returns false if the point is now outside the dimensions
// if the point becomes out of bounds, it reduces it until it it just
// outside of bounds -- equal to len(...)
func (m *MultiDim) IncreaseBy(point []int, change int) bool {
	c := m.Point(change)
	for i := range point {
		point[i] += c[i]
	}

	for i := range point {
		if point[i] > m.Dims[i] {
			if i < len(point)-1 {
				point[i+1] += point[i] / m.Dims[i]
				point[i] %= m.Dims[i]
			} else {
				point[i] = m.Dims[i]
				return false
			}
		}
	}

	return true
}
