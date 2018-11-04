package operators

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

type tanh int8

// Tanh returns an Operator that performs an element-wise application of
// the tanh() function.
func Tanh() tanh {
	return tanh(0)
}

func (t tanh) TypeString() string {
	return "tanh"
}

func (t tanh) Finalize(n *bs.Node) error {
	return nil
}

func (t tanh) Value(in float64, index int) float64 {
	return math.Tanh(in)
}

func (t tanh) Deriv(n *bs.Node, index int) float64 {
	// it's cheaper to multiply it by itself than to use math.Pow()
	return 1 - (n.Value(index) * n.Value(index))
}
