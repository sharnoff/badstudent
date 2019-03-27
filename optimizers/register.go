package optimizers

import bs "github.com/sharnoff/badstudent"

func init() {
	list := []interface{}{
		func() bs.Optimizer { return SGD() },
	}

	if err := bs.RegisterAll(list); err != nil {
		panic(err)
	}

	bs.SetDefaultOptimizer(func() bs.Optimizer{ return SGD() })
}
