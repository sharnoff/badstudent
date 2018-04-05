package badstudent

import (
	// "github.com/pkg/errors"
	"math"
	// "fmt"
)

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
