package hyperparams

import (
	bs "github.com/sharnoff/badstudent"
)

func init() {
	list := map[string]func() bs.HyperParameter{
		Constant(0).TypeString(): func() bs.HyperParameter { return Constant(0) }, // 0 is just random. It'll be loaded.
	}

	for s, f := range list {
		err := bs.RegisterHyperParameter(s, f)
		if err != nil {
			panic(err.Error())
		}
	}
}
