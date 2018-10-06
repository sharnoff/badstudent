package optimizers

import bs "github.com/sharnoff/badstudent"

func init() {
	list := map[string]func() bs.Optimizer{
		"SGD": func() bs.Optimizer { return GradientDescent() },
	}

	for s, f := range list {
		err := bs.RegisterOptimizer(s, f)
		if err != nil {
			panic(err.Error())
		}
	}

	bs.SetDefaultOptimizer(func() bs.Optimizer { return GradientDescent() })
}
