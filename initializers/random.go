package initializers

import (
	bs "github.com/sharnoff/badstudent"
)

type random struct {
	RNG
}

// Random returns an Initializer that uses the provided RNG to generate the weights. There is no
// scaling beyond that of the RNG.
func Random(g RNG) random {
	return random{g}
}

// Set is the implementation of badstudent.Initializer
func (r random) Set(n *bs.Node, ws []float64) {
	for i := 0; i < len(ws); i++ {
		ws[i] = r.Gen()
	}
}
