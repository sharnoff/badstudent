package costfuncs

import (
	bs "github.com/sharnoff/badstudent"
)

func init() {
	list := map[string]func() bs.CostFunction{
		CrossEntropy().TypeString(): func() bs.CostFunction { return CrossEntropy() },
		Huber(0).TypeString(): func() bs.CostFunction { return Huber(0) },
		MSE().TypeString(): func() bs.CostFunction { return MSE() },
		Abs().TypeString(): func() bs.CostFunction { return Abs() },
	}

	for s, f := range list {
		err := bs.RegisterCostFunction(s, f)
		if err != nil {
			panic(err.Error())
		}
	}
}
