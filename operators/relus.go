// relus.go contains all activation functions that are derivative of relu:
// * ReLU
// * Leaky ReLU
// * Parametric ReLU
// * ELU
// * Softplus (because it's similar)
package operators

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

// ****************************************
// ReLU
// ****************************************

type relu int8

// ReLU returns the standard rectified linear unit, which implements
// badstudent.Operator.
func ReLU() relu {
	return relu(0)
}

func (t relu) TypeString() string {
	return "relu"
}

func (t relu) Finalize(n *bs.Node) error {
	// We don't need to check number of values because badstudent main does it for us
	return nil
}

func (t relu) Value(in float64, index int) float64 {
	return math.Max(in, 0)
}

func (t relu) Deriv(n *bs.Node, index int) float64 {
	return math.Max(n.Value(index), 0)
}

// ****************************************
// Leaky ReLU
// ****************************************

type lrelu float64

// LeakyReLU returns a standard 'leaky ReLU', where the leaky factor is given by alpha.
func LeakyReLU(alpha float64) lrelu {
	return lrelu(alpha)
}

func (t lrelu) TypeString() string {
	return "leaky-relu"
}

func (t *lrelu) Get() interface{} {
	return *t
}

func (t *lrelu) Blank() interface{} {
	return t
}

func (t lrelu) Finalize(n *bs.Node) error {
	return nil
}

func (t lrelu) Value(in float64, index int) float64 {
	if in < 0 {
		return float64(t) * in
	}
	return in
}

func (t lrelu) Deriv(n *bs.Node, index int) float64 {
	if n.InputValue(index) < 0 {
		return float64(t)
	}
	return 1
}

// ****************************************
// PReLU
// ****************************************

type prelu struct {
	Ws []float64
}

// PReLU (parametric ReLU) returns a parameterized version of the leaky ReLU.
func PReLU() *prelu {
	return &prelu{}
}

func (t *prelu) TypeString() string {
	return "prelu"
}

func (t *prelu) Finalize(n *bs.Node) error {
	t.Ws = make([]float64, n.Size())
	return nil
}

func (t *prelu) Get() interface{} {
	return *t
}

func (t *prelu) Blank() interface{} {
	return t
}

func (t *prelu) Value(in float64, index int) float64 {
	if in < 0 {
		return t.Ws[index] * in
	}
	return in
}

func (t *prelu) Deriv(n *bs.Node, index int) float64 {
	if n.InputValue(index) < 0 {
		return t.Ws[index]
	}
	return 1
}

func (t *prelu) Weights() []float64 {
	return t.Ws
}

func (t *prelu) Grad(n *bs.Node, index int) float64 {
	if in := n.InputValue(index); in < 0 {
		return in
	}

	return 0
}

// ****************************************
// ELU
// ****************************************

type elu int8

// ELU (exponential linear unit) returns a smooth approximation of ReLU that tends towards -1 as
// inputs become infinitely negative.
func ELU() elu {
	return elu(0)
}

func (t elu) TypeString() string {
	return "elu"
}

func (t elu) Finalize(n *bs.Node) error {
	return nil
}

func (t elu) Value(in float64, index int) float64 {
	if in >= 0 {
		return in
	}
	return math.Exp(in) - 1
}

func (t elu) Deriv(n *bs.Node, index int) float64 {
	if in := n.InputValue(index); in < 0 {
		return math.Exp(in)
	}
	return 1
}

// ****************************************
// Softplus
// ****************************************

type softplus int8

// Softplus is a smooth approximation of ReLU that approaches 0 as inputs tend towards negative
// infinity.
func Softplus() softplus {
	return softplus(0)
}

func (t softplus) TypeString() string {
	return "softplus"
}

func (t softplus) Finalize(n *bs.Node) error {
	return nil
}

func (t softplus) Value(in float64, index int) float64 {
	return math.Log(1 + math.Exp(in))
}

func (t softplus) Deriv(n *bs.Node, index int) float64 {
	// 1 / (1 + e^-x)
	return 1.0 / (1 + math.Exp(-n.InputValue(index)))
}
