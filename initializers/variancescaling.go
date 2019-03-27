package initializers

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

type varianceScaling struct {
	// either: "in", "out", "avg"
	mode   string
	factor float64
}

const defaultVarianceMode string = "avg"

// VarianceScaling returns the variance scaling initializer, which has 3 modes and a user-defined
// scaling factor. The three modes can be set by In, Out, and Avg. It defaults to Avg.
func VarianceScaling() *varianceScaling {
	return &varianceScaling{defaultVarianceMode, defaultValue["varscl-factor"]}
}

// Factor sets the scaling factor to be used for the Initializer. The default factor can be set by
// SetDefault("varscl-factor")
func (v *varianceScaling) Factor(f float64) *varianceScaling {
	v.factor = f
	return v
}

// In sets the scaling to be based on the number of input values to the Node.
func (v *varianceScaling) In() *varianceScaling {
	v.mode = "in"
	return v
}

// Out sets the scaling to be based on the number of output values to the Node.
func (v *varianceScaling) Out() *varianceScaling {
	v.mode = "out"
	return v
}

// Avg sets the scaling to be based on the average of the numbers of input and output values to the
// Node.
func (v *varianceScaling) Avg() *varianceScaling {
	v.mode = "avg"
	return v
}

// Set is the implementation of badstudent.Initializer
func (v *varianceScaling) Set(n *bs.Node, ws []float64) {
	var scale float64
	if v.mode == "in" {
		scale = float64(n.NumInputs())
	} else if v.mode == "out" {
		scale = float64(n.Size())
	} else { // must be "avg"
		scale = float64(n.NumInputs()+n.Size()) / 2
	}

	gen := TruncNormal().SD(math.Sqrt(v.factor / scale))

	for i := 0; i < len(ws); i++ {
		ws[i] = gen.Gen()
	}
}
