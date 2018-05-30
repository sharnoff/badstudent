package operators

import (
	"github.com/pkg/errors"
	"github.com/sharnoff/badstudent"
	"math/rand"

	"os"
	"encoding/json"
)

type neurons struct {
	opt Optimizer

	weights [][]float64
	biases  []float64

	weightChanges [][]float64
	biasChanges   []float64
}

func Neurons(opt Optimizer) *neurons {
	n := new(neurons)
	n.opt = opt
	return n
}

const bias_value float64 = 1

func (n *neurons) Init(l *badstudent.Layer) error {

	n.weights = make([][]float64, l.Size())
	n.weightChanges = make([][]float64, l.Size())
	n.biases = make([]float64, l.Size())
	n.biasChanges = make([]float64, l.Size())

	for v := range n.weights {
		n.weights[v] = make([]float64, l.NumInputs())
		n.weightChanges[v] = make([]float64, l.NumInputs())
		for i := range n.weights[v] {
			n.weights[v][i] = (2*rand.Float64() - 1) / float64(l.NumInputs())
		}
	}

	if l.NumInputs() != 0 {
		for v := range n.biases {
			n.biases[v] = (2*rand.Float64() - 1) / float64(l.NumInputs())
		}
	}

	return nil
}

// stores the states of 'weights' and 'biases' into a single file, 'weights.txt'
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
		j := struct {
			Weights [][]float64
			Biases []float64
		}{n.weights, n.biases}

		enc := json.NewEncoder(f)
		if err = enc.Encode(&j); err != nil {
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
		var j struct {
			Weights [][]float64
			Biases []float64
		}

		dec := json.NewDecoder(f)
		if err = dec.Decode(&j); err != nil {
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

		n.weights = j.Weights
		n.biases = j.Biases

		n.weightChanges = make([][]float64, len(n.weights))
		for i := range n.weightChanges {
			n.weightChanges[i] = make([]float64, len(n.weights[i]))
		}
		n.biasChanges = make([]float64, len(n.biases))
	}

	err = n.opt.Load(l, n, dirPath + "/opt", aux)
	if err = n.opt.Load(l, n, dirPath + "/opt", aux); err != nil {
		return errors.Wrapf(err, "Couldn't load optimizer after loading operator\n")
	}

	return nil
}

func (n *neurons) Evaluate(l *badstudent.Layer, values []float64) error {

	inputs := l.CopyOfInputs()
	calculateValue := func(sl []int) {
		i := sl[0]
		var sum float64
		for in := range inputs {
			sum += n.weights[i][in] * inputs[in]
		}

		values[i] = sum + (n.biases[i] * bias_value)
	}

	opsPerThread, threadsPerCPU := 1, 1

	bounds := [][]int{[]int{0, len(values)}}
	badstudent.MultiThread(bounds, calculateValue, opsPerThread, threadsPerCPU)

	return nil
}

// used for InputDeltas()
func (n *neurons) calculateDelta(l *badstudent.Layer, add func(int, float64), index int) {
	var sum float64
	for v := 0; v < l.Size(); v++ {
		sum += l.Delta(v) * n.weights[v][index]
	}

	add(index, sum)
}

func (n *neurons) InputDeltas(l *badstudent.Layer, add func(int, float64), start, end int) error {

	sendDelta := func(sl []int) {
		i := sl[0]
		var sum float64
		for v := 0; v < l.Size(); v++ {
			sum += l.Delta(v) * n.weights[v][i]
		}

		add(i - start, sum)
	}

	opsPerThread, threadsPerCPU := 1, 1

	bounds := [][]int{[]int{start, end}}
	badstudent.MultiThread(bounds, sendDelta, opsPerThread, threadsPerCPU)

	return nil
}

func (n *neurons) CanBeAdjusted(l *badstudent.Layer) bool {
	return (len(n.weights[0]) != 0)
}

func (n *neurons) Adjust(l *badstudent.Layer, learningRate float64, saveChanges bool) error {
	inputs := l.CopyOfInputs()

	if len(inputs) == 0 {
		return nil
	}

	targetWeights := n.weightChanges
	targetBiases := n.biasChanges
	if !saveChanges {
		targetWeights = n.weights
		targetBiases = n.biases
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
	for v := range n.weights {
		for in := range n.weights[v] {
			n.weights[v][in] += n.weightChanges[v][in]
		}
		n.biases[v] += n.biasChanges[v]

		n.weightChanges[v] = make([]float64, len(n.weights[v]))
	}
	n.biasChanges = make([]float64, len(n.biases))

	return nil
}
