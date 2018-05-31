package badstudent

import (
	"runtime"
	"sync"
)

// should be run sequentially, not in a separate thread
// mainly for use by operators or optimizers in their computationally expensive calculations
//
// f is the function that should be run in parallel, where its arguments are in the range of bounds [start, end)
// -- will be supplied a slice with len equal to len(bounds)
// bounds: WYSIWYG. Should consist of an n-length slice of 2-length slices (lower and upper bounds)
// -- assumes that bounds[n][0] < bounds[n][1]
//
// counts such that bounds[0] loops once, bounds[1] loops some finite number of times, bounds[2] loops more, etc.
func MultiThread(bounds [][]int, f func([]int), opsPerThread, threadsPerCPU int) {

	numThreads := runtime.NumCPU() * threadsPerCPU

	dims := make([]int, len(bounds))
	for i := range dims {
		dims[i] = bounds[i][1] - bounds[i][0]
	}

	multiDim := NewMultiDim(dims)
	places := make([]int, len(bounds))

	var placeMux sync.Mutex

	done := make(chan bool)

	for thread := 0; thread < numThreads; thread++ {
		go func() {
			for {
				placeMux.Lock()

				done := true
				for i := range places {
					if places[i] < dims[i] {
						done = false
						break
					}
				}
				if done {
					placeMux.Unlock()
					break
				}

				p := make([]int, len(places))
				copy(p, places)

				change := opsPerThread
				// leaves 'places' at the maximum value if it goes over
				multiDim.IncreaseBy(places, change)

				e := make([]int, len(places))
				copy(e, places)

				placeMux.Unlock()

				// the actual places within 'bounds' that 'places' correlates to
				temp := make([]int, len(places))

				for {
					// if p == e, quit
					different := false
					for i := len(p) - 1; i >= 0; i-- {
						if p[i] != e[i] {
							different = true
							break
						}
					}
					if !different {
						break
					}

					for i := range temp {
						temp[i] = bounds[i][0] + p[i]
					}

					f(temp)

					// p++
					multiDim.Increment(p)
				}
			}

			done <- true
		}()
	}

	numFinished := 0
	for numFinished < numThreads {
		<-done
		numFinished++
	}

	return
}

// allows storage of n-dimensional slices
//
// stored such that the oscillation frequency of the dimensions decreases as
// the index in dimensions increases
//
// primarily used in n-dimensional range multi-threading,
// and is made public for use in mult-dimensional operators
//
// the fields are made public in order to allow exporting to JSON,
// but they should not actually be altered once it has been initialized
type MultiDim struct {
	// the dimensions of the 
	Dims []int

	// the number of values encapsulated by a 'set' of this dimension
	// -- size[0] = dims[0]; size[end] = len(base)
	// does not need to 
	Size []int
}

// assumes that the product of 'dims' multiplied together is equal to the length of 'base'
// stored as: z{y{x, x}, y{x, x}}, z{y{x, x}, y{x, x}}.
// accessed as: [x, y, z]
func NewMultiDim(dims []int) *MultiDim {
	m := &MultiDim{
		Dims: dims,
		Size: make([]int, len(dims)),
	}

	m.Size[0] = m.Dims[0]

	for i := 1; i < len(m.Size); i++ {
		m.Size[i] = m.Size[i - 1] * m.Dims[i]
	}

	return m
}

// returns the index corresponding to the given point
// assumes that the point has the same number of dimensions as 'm'
//
// the given point should have the dimensions in the same order as they were originally given
func (m *MultiDim) Index(point []int) int {
	index := 0
	for i := range m.Size {
		index += point[i] * m.Size[i]
	}

	return index
}

// returns the multi-dimensional point leading to the given index in the base array
func (m *MultiDim) Point(index int) []int {
	p := make([]int, len(m.Dims))
	for i := len(p) - 1; i >= 1; i-- { // doesn't go to 0
		p[i] = index / m.Size[i - 1]
		index = index % m.Size[i - 1]
	}

	p[0] = index
	return p
}

// increments the given point by 1
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

		if i != len(point) - 1 {
			point[i] = 0
		} else {
			return false
		}
	}

	return true
}

// increments the point by the amount specified
// cannnot accept negative numbers
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
			if i < len(point) - 1 {
				point[i + 1] += point[i] / m.Dims[i]
				point[i] %= m.Dims[i]
			} else {
				point[i] = m.Dims[i]
				return false
			}
		}
	}

	return true
}
