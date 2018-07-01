package utils

import (
	"runtime"
	"sync"
)

// Multithreads an operation on a range of integers
//
// should be run sequentially, not in a separate thread
// designed for use by operators or optimizers in their mass calculations
//
// the range includes 'start' and excludes 'end'
//  - MultiThread assumes that end ≥ start
// 'f' is the function that should be run for each value in the range
// 'opsPerThread' is the number of operations that each goroutine will handle before requesting another set
// 'threadsPerCPU' is the number of goroutines created for each CPU
func MultiThread(start, end int, f func(int), opsPerThread, threadsPerCPU int) {

	numThreads := runtime.NumCPU() * threadsPerCPU
	index := start
	var indexMux sync.Mutex

	var wg sync.WaitGroup

	wg.Add(numThreads)
	for thread := 0; thread < numThreads; thread++ {
		go func() {
			for {

				indexMux.Lock()
				if index >= end {
					indexMux.Unlock()
					break
				}

				i := index
				index += opsPerThread
				indexMux.Unlock()

				e := i + opsPerThread

				if e > end {
					e = end
				}

				for ; i < e; i++ {
					f(i)
				}
			}

			wg.Done()
		}()
	}

	wg.Wait()

	return
}
