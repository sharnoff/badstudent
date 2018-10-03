package operators

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/optimizers"
)

func init() {
	list := map[string]func() bs.Operator {
		"add": func() bs.Operator {return Add()},
		"convolution": func() bs.Operator {return Convolution(nil, optimizers.GradientDescent())},
		"identity": func() bs.Operator {return Identity()},
		"logistic": func() bs.Operator {return Logistic()},
		"multiply": func() bs.Operator {return Mult()},
		"neurons": func() bs.Operator {return Neurons(optimizers.GradientDescent())},
		"avg-pool": func() bs.Operator {return AvgPool(nil)},
		"max-pool": func() bs.Operator {return MaxPool(nil)},
		"tanh": func() bs.Operator {return Tanh()},
	}

	for s, f := range list {
		err := bs.RegisterOperator(s, f)
		if err != nil {
			panic(err.Error())
		}
	}
}