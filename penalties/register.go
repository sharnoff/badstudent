package penalties

import bs "github.com/sharnoff/badstudent"

func init() {
	list := []interface{}{
		func() bs.Penalty { return ElasticNet(0,0) },
		func() bs.Penalty { return L1(0) },
		func() bs.Penalty { return L2(0) },
	}

	if err := bs.RegisterAll(list); err != nil {
		panic(err)
	}
}