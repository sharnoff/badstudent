package operators

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

// ****************************************
// Logistic
// ****************************************

type logistic int8

// Logistic returns an elementwise application of the logistic (or sigmoid) function that
// implements badstudent.Operator.
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

// ****************************************
// Tanh
// ****************************************

type tanh int8

// Tanh returns an Operator that performs an element-wise application of the tanh() function.
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

// ****************************************
// Softsign
// ****************************************

type softsign int8

// Softsign (not to be confused with softplus) returns the Softsign activation function. It is
// similar in shape to Tanh and Logistic.
func Softsign() softsign {
	return softsign(0)
}

func (t softsign) TypeString() string {
	return "softsign"
}

func (t softsign) Finalize(n *bs.Node) error {
	return nil
}

func (t softsign) Value(in float64, index int) float64 {
	return in / (math.Abs(in) + 1)
}

func (t softsign) Deriv(n *bs.Node, index int) float64 {
	// 1 / (|in| + 1)^2
	return math.Pow(math.Abs(n.InputValue(index))+1, 2)
}
