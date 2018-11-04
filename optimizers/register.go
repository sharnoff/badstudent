package optimizers

import bs "github.com/sharnoff/badstudent"

func init() {
	list := map[string]func() bs.Optimizer{
		SGD().TypeString(): func() bs.Optimizer { return SGD() },
	}

	for s, f := range list {
		err := bs.RegisterOptimizer(s, f)
		if err != nil {
			panic(err.Error())
		}
	}

	bs.SetDefaultOptimizer(list[SGD().TypeString()])
}
