package operators

import (
	"github.com/pkg/errors"
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/utils"

	"math/rand"

	"encoding/json"
	"os"
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

func (n *neurons) Init(nd *bs.Node) error {

	if nd.NumInputs() == 0 {
		return errors.Errorf("Neurons must have inputs (NumInputs() == %d)", nd.NumInputs())
	}

	n.Weights = make([][]float64, nd.Size())
	n.WeightChanges = make([][]float64, nd.Size())
	n.Biases = make([]float64, nd.Size())
	n.BiasChanges = make([]float64, nd.Size())

	for v := range n.Weights {
		n.Weights[v] = make([]float64, nd.NumInputs())
		n.WeightChanges[v] = make([]float64, nd.NumInputs())
		for i := range n.Weights[v] {
			n.Weights[v][i] = (2*rand.Float64() - 1) / float64(nd.NumInputs())
		}
	}

	if nd.NumInputs() != 0 {
		for v := range n.Biases {
			n.Biases[v] = (2*rand.Float64() - 1) / float64(nd.NumInputs())
		}
	}

	return nil
}

// encodes 'n' via JSON into 'weights.txt'
func (n *neurons) Save(nd *bs.Node, dirPath string) error {
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

	if err = n.opt.Save(nd, n, dirPath+"/opt"); err != nil {
		return errors.Wrapf(err, "Couldn't save optimizer after saving operator")
	}

	return nil
}

// decodes JSON from 'weights.txt'
func (n *neurons) Load(nd *bs.Node, dirPath string, aux []interface{}) error {

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

		if nd.Size() != len(n.Weights) || nd.Size() != len(n.Biases) {
			return errors.Errorf("Couldn't load operator: !(nd.Size() == len(weights) == len(biases)) (%d, %d, %d)", nd.Size(), len(n.Weights), len(n.Biases))
		}
		numInputs := nd.NumInputs()
		for i := range n.Weights {
			if numInputs != len(n.Weights[i]) {
				return errors.Errorf("Couldn't load operator: nd.NumInputs() != len(weights[%d]) (%d != %d)", i, numInputs, len(n.Weights[i]))
			}
		}
	}

	if err = n.opt.Load(nd, n, dirPath+"/opt", aux); err != nil {
		return errors.Wrapf(err, "Couldn't load optimizer after loading operator\n")
	}

	return nil
}

func (n *neurons) Evaluate(nd *bs.Node, values []float64) error {

	inputs := nd.CopyOfInputs()
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

func (n *neurons) Value(nd *bs.Node, index int) float64 {
	var sum float64
	var in int
	for inv := range nd.InputIterator() {
		sum += n.Weights[index][in] * inv
		in++
	}

	return sum + (n.Biases[index] * bias_value)
}

func (n *neurons) InputDeltas(nd *bs.Node, add func(int, float64), start, end int) error {

	sendDelta := func(i int) {
		var sum float64
		for v := 0; v < nd.Size(); v++ {
			sum += nd.Delta(v) * n.Weights[v][i]
		}

		add(i-start, sum)
	}

	opsPerThread, threadsPerCPU := 1, 1

	utils.MultiThread(start, end, sendDelta, opsPerThread, threadsPerCPU)

	return nil
}

func (n *neurons) CanBeAdjusted(nd *bs.Node) bool {
	return true
}

func (n *neurons) NeedsValues(nd *bs.Node) bool {
	return false
}

func (n *neurons) NeedsInputs(nd *bs.Node) bool {
	return true
}

func (n *neurons) Adjust(nd *bs.Node, learningRate float64, saveChanges bool) error {
	inputs := nd.CopyOfInputs()

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

			return inputs[in] * nd.Delta(v)
		}

		add := func(index int, addend float64) {
			in := index % len(inputs)
			v := (index - in) / len(inputs)

			targetWeights[v][in] += addend
		}

		if err := n.opt.Run(nd, len(inputs)*nd.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust node %v, running optimizer on weights failed\n", nd)
		}
	}

	// now run on biases
	{
		grad := func(index int) float64 {
			return bias_value * nd.Delta(index)
		}

		add := func(index int, addend float64) {
			targetBiases[index] += addend
		}

		if err := n.opt.Run(nd, nd.Size(), grad, add, learningRate); err != nil {
			return errors.Wrapf(err, "Couldn't adjust node %v, running optimizer on biases failed\n", nd)
		}
	}

	return nil
}

func (n *neurons) AddWeights(nd *bs.Node) error {
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
