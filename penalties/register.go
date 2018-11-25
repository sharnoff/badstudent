package penalties

import bs "github.com/sharnoff/badstudent"

func init() {
	list := map[string]func() bs.Penalty {
		ElasticNet(0,0).TypeString(): func() bs.Penalty { return ElasticNet(0,0) },
		L1(0).TypeString(): func() bs.Penalty { return L1(0) },
		L2(0).TypeString(): func() bs.Penalty { return L2(0) },
	}

	for s, f := range list {
		err := bs.RegisterPenalty(s, f)
		if err != nil {
			panic(err.Error())
		}
	}
}