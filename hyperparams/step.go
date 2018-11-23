package hyperparams

import (
	"encoding/json"
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"os"
)

type step struct {
	Iter int
	Val  float64
}

type stepper []step

func Step(base float64) *stepper {
	s := make([]step, 1)

	s[0] = step{0, base}

	st := stepper(s)
	return &st
}

// Add adds a step to the HyperParameter.
func (s *stepper) Add(iter int, value float64) *stepper {
	*s = append(*s, step{iter, value})
	return s
}

func (s *stepper) TypeString() string {
	return "step"
}

func (s *stepper) Value(iter int) float64 {
	sl := []step(*s)
	for i := 1; i < len(sl); i++ {
		if sl[i].Iter > iter {
			return sl[i-1].Val
		}
	}

	return sl[len(sl)-1].Val
}

func (s *stepper) Save(n *bs.Node, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Failed to create directory %q", dirPath)
	}

	f, err := os.Create(dirPath + "/stepper.txt")
	if err != nil {
		return errors.Errorf("Failed to create file %q in %q", "stepper.txt", dirPath)
	}

	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(s); err != nil {
		return errors.Errorf("Failed to encode JSON to file %q in %q", "stepper.txt", dirPath)
	}

	return nil
}

func (s *stepper) Load(dirPath string) error {
	f, err := os.Open(dirPath + "/stepper.txt")
	if err != nil {
		return errors.Errorf("Failed to open file %q in %q", "stepper.txt", dirPath)
	}

	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(s); err != nil {
		return errors.Wrapf(err, "Failed to decode JSON from file %q in %q", "stepper.txt", dirPath)
	}

	return nil
}
