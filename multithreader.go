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

	places := make([]int, len(bounds))
	sizes := make([]int, len(bounds))
	for i := range bounds {
		places[i] = bounds[i][0]
		sizes[i] = bounds[i][1] - bounds[i][0]
	}

	for i := len(sizes) - 2; i >= 0; i-- {
		sizes[i] = sizes[i] * sizes[i + 1]
	}

	var placeMux sync.Mutex

	done := make(chan bool)

	for thread := 0; thread < numThreads; thread++ {
		go func() {
			for {

				placeMux.Lock()
				done := true
				for i := range places {
					if places[i] < bounds[i][1] {
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

				// update places to all of the new indexes that it should be at,
				// according to the total change.
				//
				// essentially breaks down the change 
				for i := range sizes {
					places[i] += change / sizes[i]
					change = change % sizes[i]
				}

				e := make([]int, len(places))
				copy(e, places)

				placeMux.Unlock()

				// none of the other ones should go over
				if e[0] > bounds[0][1] {
					e[0] = bounds[0][1]
				}

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

					f(p)

					// p++
					for i := len(p) - 1; i >= 0; i-- {
						p[i]++
						if p[i] < bounds[i][1] {
							break
						}

						if i != 0 {
							p[i] = bounds[i][0]
						}
					}
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
