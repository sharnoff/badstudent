package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"
	"math/rand"

	"os"
	"encoding/json"
)

type neurons struct {
	opt Optimizer

	Weights [][]float64
	Biases  []float64

	WeightChanges [][]float64
	BiasChanges   []float64
}

func Neurons(opt Optimizer) *neurons {
	n := new(neurons)
	n.opt = opt
	return n
}

const bias_value float64 = 1

func (n *neurons) Init(l *badstudent.Layer) error {

	n.Weights = make([][]float64, l.Size())
	n.WeightChanges = make([][]float64, l.Size())
	n.Biases = make([]float64, l.Size())
	n.BiasChanges = make([]float64, l.Size())

	for v := range n.Weights {
		n.Weights[v] = make([]float64, l.NumInputs())
		n.WeightChanges[v] = make([]float64, l.NumInputs())
		for i := range n.Weights[v] {
			n.Weights[v][i] = (2*rand.Float64() - 1) / float64(l.NumInputs())
		}
	}

	if l.NumInputs() != 0 {
		for v := range n.Biases {
			n.Biases[v] = (2*rand.Float64() - 1) / float64(l.NumInputs())
		}
	}

	return nil
}

// encodes 'n' via JSON into 'weights.txt'
func (n *neurons) Save(l *badstudent.Layer, dirPath string) error {
	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Errorf("Couldn't save operator: failed to create directory to house save file")
	}

	f, err := os.Create(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't save operator: failed to create file 'weights.txt'")
	}

	finishedSafely := false
	defer func() {
		if !finishedSafely {
			f.Close()
		}
	}()

	{
		enc := json.NewEncoder(f)
		if err = enc.Encode(n); err != nil {
			return errors.Wrapf(err, "Couldn't save operator: failed to encode JSON to file\n")
		}
		finishedSafely = true
		f.Close()
	}

	if err = n.opt.Save(l, n, dirPath + "/opt"); err != nil {
		return errors.Wrapf(err, "Couldn't save optimizer after saving operator")
	}

	return nil
}

// decodes JSON from 'weights.txt'
func (n *neurons) Load(l *badstudent.Layer, dirPath string, aux []interface{}) error {

	f, err := os.Open(dirPath + "/weights.txt")
	if err != nil {
		return errors.Errorf("Couldn't load operator: could not open file 'weights.txt'")
	}

	finishedSafely := false
	defer func() {
		if !finishedSafely {
			f.Close()
		}
	}()

	{
		dec := json.NewDecoder(f)
		if err = dec.Decode(n); err != nil {
			return errors.Wrapf(err, "Couldn't load operator: failed to decode JSON from file\n")
		}
		finishedSafely = true
		f.Close()

		if l.Size() != len(j.Weights) || l.Size() != len(j.Biases) {
			return errors.Errorf("Couldn't load operator: !(l.Size() == len(weights) == len(biases)) (%d, %d, %d)", l.Size(), len(j.Weights), len(j.Biases))
		}
		numInputs := l.NumInputs()
		for i := range j.Weights {
			if numInputs != len(j.Weights[i]) {
				return errors.Errorf("Couldn't load operator: l.NumInputs() != len(weights[%d]) (%d != %d)", i, numInputs, len(j.Weights[i]))
			}
		}
	}

	if err = n.opt.Load(l, n, dirPath + "/opt", aux); err != nil {
		return errors.Wrapf(err, "Couldn't load optimizer after loading operator\n")
	}

	return nil
}

func (n *neurons) Evaluate(l *badstudent.Layer, values []float64) error {

	inputs := l.CopyOfInputs()
	calculateValue := func(i int) {
		var sum float64
		for in := range inputs {
			sum += n.Weights[i][in] * inputs[in]
		}

		values[i] = sum + (n.Biases[i] * bias_value)
	}

	opsPerThread, threadsPerCPU := 1, 1
	utils.MultiThread(0, len(values), calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

func (n *neurons) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

	sendDelta := func(i int) {
		var sum float64
		for v := 0; v < l.Size(); v++ {
			sum += l.Delta(v) * n.Weights[v][i]
		}

		add(i - start, sum)
	}

	opsPerThread, threadsPerCPU := 1, 1

	utils.MultiThread(start, end, sendDelta, opsPerThread, threadsPerCPU)

	return nil
}

func (n *neurons) CanBeAdjusted(l *badstudent.Layer) bool {
	return (len(n.Weights[0]) != 0)
}

func (n *neurons) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	inputs := l.CopyOfInputs()

	if len(inputs) == 0 {
		return nil
	}

	targetWeights := n.WeightChanges
	targetBiases := n.BiasChanges
	if !saveChanges {
		targetWeights = n.Weights
		targetBiases = n.Biases
	}

	// first run on weights, then biases
	{
		grad := func(index int) float64 {
			in := index % len(inputs)
			v := (index - in) / len(inputs)

			return inputs[in] * l.Delta(v)
		}

		add := func(index int, addend float64) {
			in := index % len(inputs)
			v := (index - in) / len(inputs)

			targetWeights[v][in] += addend
		}

		if err := n.opt.Run(l, len(inputs)*l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on weights failed\n", l)
		}
	}

	// now run on biases
	{
		grad := func(index int) float64 {
			return bias_value * l.Delta(index)
		}

		add := func(index int, addend float64) {
			targetBiases[index] += addend
		}

		if err := n.opt.Run(l, l.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust layer %v, running optimizer on biases failed\n", l)
		}
	}

	return nil
}

func (n *neurons) AddWeights(l *badstudent.Layer) error {
	for v := range n.Weights {
		for in := range n.Weights[v] {
			n.Weights[v][in] += n.WeightChanges[v][in]
		}
		n.Biases[v] += n.BiasChanges[v]

		n.WeightChanges[v] = make([]float64, len(n.Weights[v]))
	}
	n.BiasChanges = make([]float64, len(n.Biases))

	return nil
}
