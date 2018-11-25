package penalties

import (
	bs "github.com/sharnoff/badstudent"
	"math"
)

// **********************************************
// L1 (Lasso)
// **********************************************

type l1 float64

// λ is a small value close to 0 where λ > 0
func L1(λ float64) *l1 {
	p := l1(λ)
	return &p
}

// λ is a small value close to 0 where λ > 0
func Lasso(λ float64) *l1 {
	return L1(λ)
}

func (p *l1) TypeString() string {
	return "l1-lasso"
}

func (p *l1) Penalize(n *bs.Node, adj bs.Adjustable, index int) float64 {
	λ := float64(*p)
	w := adj.Weights()[index]
	return adj.Grad(n, index) + λ * math.Copysign(1, w)
}

func (p *l1) Get() interface{} {
	return *p
}

func (p *l1) Blank() interface{} {
	return p
}

// **********************************************
// L2 (Ridge)
// **********************************************

type l2 float64

// λ is a small value close to 0 where λ > 0
func L2(λ float64) *l2 {
	p := l2(λ)
	return &p
}

// λ is a small value close to 0 where λ > 0
func Ridge(λ float64) *l2 {
	return L2(λ)
}

func (p *l2) TypeString() string {
	return "l2-ridge"
}

func (p *l2) Penalize(n *bs.Node, adj bs.Adjustable, index int) float64 {
	λ := float64(*p)
	w := adj.Weights()[index]
	return adj.Grad(n, index) + 2*λ*w
}

func (p *l2) Get() interface{} {
	return *p
}

func (p *l2) Blank() interface{} {
	return p
}
