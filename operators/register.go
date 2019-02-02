package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"math"
)

func init() {
	list := []interface{}{
		func() bs.Operator { return LeakyReLU(0) },
		func() bs.Operator { return Identity() },
		func() bs.Operator { return Logistic() },
		func() bs.Operator { return Softplus() },
		func() bs.Operator { return Softsign() },
		func() bs.Operator { return AvgPool() },
		func() bs.Operator { return MaxPool() },
		func() bs.Operator { return Neurons() },
		func() bs.Operator { return Softmax() },
		func() bs.Operator { return PReLU() },
		func() bs.Operator { return Conv() },
		func() bs.Operator { return Mult() },
		func() bs.Operator { return Tanh() },
		func() bs.Operator { return ReLU() },
		func() bs.Operator { return ELU() },
		func() bs.Operator { return Add() },
	}

	if err := bs.RegisterAll(list); err != nil {
		panic(err)
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
