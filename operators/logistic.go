package operators

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

type logistic int8

// Logistic returns an elementwise application of the logistic (or sigmoid) function
// that implements badstudent.Operator.
func Logistic() logistic {
	return logistic(0)
}

func (t logistic) TypeString() string {
	return "logistic"
}

func (t logistic) Finalize(n *bs.Node) error {
	// We don't need to check number of values because badstudent main does it for
	// us
	return nil
}

func (t logistic) Value(in float64, index int) float64 {
	// the logistic function can be rephrased as:
	return 0.5 + 0.5*math.Tanh(0.5*in)
}

func (t logistic) Deriv(n *bs.Node, index int) float64 {
	return n.Value(index) * (1 - n.Value(index))
}
