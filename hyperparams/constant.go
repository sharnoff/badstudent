package hyperparams

import (
	"encoding/json"
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"os"
)

type constant float64

func Constant(value float64) *constant {
	c := constant(value)
	return &c
}

func (c constant) TypeString() string {
	return "constant"
}

func (c *constant) Value(iter int) float64 {
	return *(*float64)(c)
}

func (c *constant) Save(n *bs.Node, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Failed to create directory %q", dirPath)
	}

	f, err := os.Create(dirPath + "/constant.txt")
	if err != nil {
		return errors.Errorf("Failed to create file %q in %q", "constant.txt", dirPath)
	}

	defer f.Close()

	v := float64(*c)

	enc := json.NewEncoder(f)
	if err = enc.Encode(v); err != nil {
		return errors.Errorf("Failed to encode JSON to file %q in %q", "constant.txt", dirPath)
	}

	return nil
}

func (c *constant) Load(dirPath string) error {
	f, err := os.Open(dirPath + "/constant.txt")
	if err != nil {
		return errors.Errorf("Failed to open file %q in %q", "constant.txt", dirPath)
	}

	defer f.Close()

	var v float64

	dec := json.NewDecoder(f)
	if err = dec.Decode(&v); err != nil {
		return errors.Wrapf(err, "Failed to decode JSON from file %q in %q", "constant.txt", dirPath)
	}

	*c = constant(v)

	return nil
}
