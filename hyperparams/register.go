package hyperparams

import bs "github.com/sharnoff/badstudent"

func init() {
	list := []interface{}{
		func() bs.HyperParameter { return Constant(0) },
		func() bs.HyperParameter { return Step(0) },
	}

	if err := bs.RegisterAll(list); err != nil {
		panic(err)
	}
}
