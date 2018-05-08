package badstudent

import (
	"runtime"
	"sync"
)

// should be run sequentially, not in a separate thread
// mainly for use by operators or optimizers in their mass calculations
// f is the function that should be run multithreaded
func MultiThread(start, end int, f func (int), opsPerThread, threadsPerCPU int) {
	
	numThreads := runtime.NumCPU() * threadsPerCPU
	index := start
	var indexMux sync.Mutex

	done := make(chan bool)

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

			done <- true
		}()
	}

	numFinished := 0
	for numFinished < numThreads {
		<- done
		numFinished++
	}

	return
}