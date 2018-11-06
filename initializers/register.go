package initializers

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"math"
)

// default values, because 'default' is a keyword
var defaultValue map[string]float64

func init() {
	bs.SetDefaultInitializer(Random(Uniform()))
	defaultValue = map[string]float64{
		"uniform-lower": -1,
		"uniform-upper": 1,
		"normal-mean":   0,
		"normal-sd":     1,
		"varscl-factor": 1,
	}
}

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
