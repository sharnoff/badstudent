package smartlearn

import (
	"github.com/pkg/errors"
	"math"
	"fmt"
)

func SquaredError(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.Errorf("Can't get squared error, len(a) != len(b) (%d != %d)", len(a), len(b))
	}

	fmt.Println(a, b)

	var sum float64
	for i := range a {
		sum += math.Pow(a[i] - b[i], 2)
	}

	return sum, nil
}

// assumes len(outs) == len(targets_)
func IsCorrect(outs, targets []float64) bool {
	for i := range outs {
		// rounds to 0 if a number is < 0.5, 1 if > 0.5. Tanh reduces the value to (0, 1)
		if math.Round(0.5 * (1 + math.Tanh(outs[i] - 0.5))) != targets[i] {
			return false
		}
	}

	return true
}
