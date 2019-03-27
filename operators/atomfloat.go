// This file is heavily based on the methods found at "go.uber.org/atomic" that pertain to float64.
// It has been stored here to ensure forward compatibility. The original source can be found at
// https://github.com/uber-go/atomic
package operators

import (
	"math"
	"sync/atomic"
	"unsafe"
)

func atomAdd(v *uint64, s float64) {
	for {
		old := atomLoad(v)
		n := old + s
		if atomCAS(v, old, n) {
			return
		}
	}
}

func atomLoad(v *uint64) float64 {
	return math.Float64frombits(atomic.LoadUint64(v))
}

func atomCAS(v *uint64, old, new float64) bool {
	return atomic.CompareAndSwapUint64(v, math.Float64bits(old), math.Float64bits(new))
}

func uint64ToFloat64(u []uint64) []float64 {
	return *(*[]float64)(unsafe.Pointer(&u))
}
