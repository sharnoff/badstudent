package costfuncs

import bs "github.com/sharnoff/badstudent"

func init() {
	list := []interface{}{
		func() bs.CostFunction { return CrossEntropy() },
		func() bs.CostFunction { return Huber(0) },
		func() bs.CostFunction { return MSE() },
		func() bs.CostFunction { return Abs() },
	}

	if err := bs.RegisterAll(list); err != nil {
		panic(err)
	}
}
