package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
)

type identity int8

// Identity returns an operator that returns its inputs
func Identity() identity {
	return identity(0)
}

func (t identity) TypeString() string {
	return "identity"
}

func (t identity) Finalize(n *bs.Node) error {
	if n.Size() != n.NumInputs() {
		return errors.Errorf("Identity Operator must have same number of values as inputs (%d != %d)", n.Size(), n.NumInputs())
	}

	return nil
}

func (t identity) Value(v float64, index int) float64 {
	return v
}

func (t identity) Deriv(n *bs.Node, index int) float64 {
	return 1
}
