package smartlearning

import (
	"github.com/pkg/errors"
	"math"
)

// returns the average squared between the two slices
func SquaredError(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.Errorf("len(a) != len(b) (%d != %d)", len(a), len(b))
	}

	sum := 0.0
	for i := range a {
		sum += math.Pow(a[i]-b[i], 2)
	}

	return sum, nil
}
