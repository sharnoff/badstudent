package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"math"
)

func init() {
	list := map[string]func() bs.Operator{
		Identity().TypeString(): func() bs.Operator { return Identity() },
		Logistic().TypeString(): func() bs.Operator { return Logistic() },
		AvgPool().TypeString():  func() bs.Operator { return AvgPool() },
		MaxPool().TypeString():  func() bs.Operator { return MaxPool() },
		Neurons().TypeString():  func() bs.Operator { return Neurons() },
		Conv().TypeString():     func() bs.Operator { return Conv() },
		Mult().TypeString():     func() bs.Operator { return Mult() },
		Tanh().TypeString():     func() bs.Operator { return Tanh() },
		Add().TypeString():      func() bs.Operator { return Add() },
	}

	for s, f := range list {
		err := bs.RegisterOperator(s, f)
		if err != nil {
			panic(err.Error())
		}
	}

	defaultValue = map[string]float64{
		"neurons-bias": 1,
		"pool-padding": 0,
		"conv-bias":    1,
		"conv-padding": 0,
	}
}

var defaultValue map[string]float64

// SetDefault sets the default values for certain Operators. The values that can be
// set are: "neurons-bias", "pool-padding", "conv-bias", and "conv-padding".
func SetDefault(name string, value float64) error {
	if _, ok := defaultValue[name]; !ok {
		return errors.Errorf("Value with name %q does not exist", name)
	} else if math.IsNaN(value) || math.IsInf(value, 0) {
		return errors.Errorf("Value is invalid (%v)", value)
	}

	return nil
}

// SetDefault_Lazy simply calls SetDefault, but panics instead of returning an error
func SetDefault_Lazy(name string, value float64) {
	if err := SetDefault(name, value); err != nil {
		panic(err)
	}
}
