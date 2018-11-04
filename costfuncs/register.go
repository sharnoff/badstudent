package costfuncs

import (
	bs "github.com/sharnoff/badstudent"
)

func init() {
	list := map[string]func() bs.CostFunction{
		MSE().TypeString(): func() bs.CostFunction { return MSE() },
	}

	for s, f := range list {
		err := bs.RegisterCostFunction(s, f)
		if err != nil {
			panic(err.Error())
		}
	}
}
