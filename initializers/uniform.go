package initializers

import (
	bs "github.com/sharnoff/badstudent"
	"math/rand"
)

type uniform struct {
	lower, upper float64
}

// Uniform returns an Initalizer that draws from a uniform random sample within a
// range, which can be set by Range. The defaults ("uniform-lower" and
// "uniform-upper") can be set by SetDefault
//
// The result of Uniform is a type that implements badstudent.Initializer.
//
// Uniform is the default Initializer
func Uniform() *uniform {
	return &uniform{defaultValue["uniform-lower"], defaultValue["uniform-upper"]}
}

// Range sets the Range of a Uniform Initializer, returning the same Initializer
func (u *uniform) Range(lower, upper float64) *uniform {
	u.lower = lower
	u.upper = upper
	return u
}

func (u *uniform) Set(n *bs.Node, ws []float64) {
	if u.lower > u.upper {
		u.lower, u.upper = u.upper, u.lower
	}

	for i := 0; i < len(ws); i++ {
		w := rand.Float64()*(u.upper-u.lower) + u.lower
		if w == 0 {
			// discard and try again
			i--
			continue
		}
		ws[i] = w
	}
}
