package penalties

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

type elasticNet struct {
	α float64
	λ float64
}

// λ is a small value close to 0 where λ > 0,
// α is a value that controls the ratio between L1 and L2
// Regularization, where 0 ≤ a ≤ 1. a = 1 is functionally identical to L1 and a = 0 is equivalent to
// L2.
func ElasticNet(α, λ float64) *elasticNet {
	return &elasticNet{α, λ}
}

func (p *elasticNet) TypeString() string {
	return "elastic-net"
}

func (p *elasticNet) Penalize(n *bs.Node, adj bs.Adjustable, index int) float64 {
	w := adj.Weights()[index]
	return adj.Grad(n, index) + p.λ * ((1 - p.α) *2*w + p.α*math.Copysign(1, w))
}

func (p *elasticNet) Get() interface{} {
	return *p
}

func (p *elasticNet) Blank() interface{} {
	return p
}
