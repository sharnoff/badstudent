package smartlearn

import (
	"github.com/pkg/errors"
	"sort"
	"math"
)

// if dupe is true, GetOutputs will return a copy of the network
// outputs, else it will return the slice of values that the network outputs
// set
func (net *Network) GetOutputs(inputs []float64, dupe bool) ([]float64, error) {
	err := net.setInputs(inputs)
	if err != nil {
		return nil, errors.Wrapf(err, "Couldn't get outputs of network, failed to first set inputs\n")
	}

	// evaluate the outputs so that we can get them from the network
	{
		results := make([]chan error, len(net.outSegments))
		for i, out := range net.outSegments {
			results[i] = make(chan error)
			go func(){ out.comLine <- commandWrapper{com: evaluate, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return nil, errors.Wrapf(err, "Couldn't get network outputs, output segment %s failed to evaluate\n", net.outSegments[i].Name)
			}
		}
	}

	// get the network values and return
	{
		if dupe {
			vals := make([]float64, len(net.outputs))
			copy(vals, net.outputs)
			return vals, nil
		} else {
			return net.outputs, nil
		}
	}
}

func (net *Network) setInputs(inputs []float64) error {
	// set the network inputs
	{
		// doesn't check if they have the same length, because copy will take care of that
		copy(net.inputs, inputs)
	}

	// notify input segments that their inputs have changed
	{
		results := make([]chan error, len(net.inSegments))
		for i, in := range net.inSegments {
			results[i] = make(chan error)
			go func(){ in.comLine <- commandWrapper{com: inputsChanged, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return errors.Wrapf(err, "Couldn't get network outputs, notifying input segment %s that inputs have changed failed\n", net.inSegments[i].Name)
			}
		}
	}

	return nil
}

// returns what the network outputs were
func (net *Network) Correct(inputs, targetOutputs []float64, learningRate float64) ([]float64, error) {
	outs, err := net.GetOutputs(inputs, true)
	if err != nil {
		return nil, errors.Wrapf(err, "Couldn't correct network, failed to get outputs\n")
	}

	// calculate deltas
	{
		results := make([]chan error, len(net.inSegments))
		for i, in := range net.inSegments {
			results[i] = make(chan error)
			go func(){ in.comLine <- commandWrapper{deltas, results[i], []interface{}{targetOutputs}} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return nil, errors.Wrapf(err, "Couldn't correct network, calculating deltas of network input %s failed\n", net.inSegments[i].Name)
			}
		}
	}

	// adjust, starting at outputs
	{
		results := make([]chan error, len(net.outSegments))
		for i, out := range net.outSegments {
			results[i] = make(chan error)
			go func(){ out.comLine <- commandWrapper{adjust, results[i], []interface{}{learningRate, targetOutputs}} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return nil, errors.Wrapf(err, "Couldn't correct network, adjusting network output %s failed\n", net.outSegments[i].Name)
			}
		}
	}

	// now that we've adjusted things, we need to tell them that their inputs have changed
	{
		results := make([]chan error, len(net.inSegments))
		for i, in := range net.inSegments {
			results[i] = make(chan error)
			go func(){ in.comLine <- commandWrapper{com: inputsChanged, res: results[i]} }()
		}
		for i, ch := range results {
			if err := <-ch; err != nil {
				return nil, errors.Wrapf(err, "Couldn't correct network, notifying that inputs changed for input %s failed\n", net.inSegments[i].Name)
			}
		}
	}

	return outs, nil
}

type arrWithIndexes struct {
	values  []float64
	indexes []int
}

func (a arrWithIndexes) Len() int {
	return len(a.values)
}
func (a arrWithIndexes) Less(i, j int) bool {
	return a.values[i] > a.values[j]
}
func (a arrWithIndexes) Swap(i, j int) {
	a.values[i], a.values[j] = a.values[j], a.values[i]
	a.indexes[i], a.indexes[j] = a.indexes[j], a.indexes[i]
}

// copies the given slice
func sortValues(vs []float64) []int {
	cop := make([]float64, len(vs))
	copy(cop, vs)
	arr := arrWithIndexes{cop, make([]int, len(cop))}
	for i := range arr.indexes {
		arr.indexes[i] = i
	}
	sort.Sort(arr)
	return arr.indexes
}

type Datum struct {
	Inputs, Outputs []float64
}

func SquaredError(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.Errorf("Can't get squared error, len(a) != len(b) (%d != %d)", len(a) != len(b))
	}

	var sum float64
	for i := range a {
		sum += math.Pow(a[i] - b[i], 2)
	}

	return sum, nil
}

// @CHANGE : allow for changing learning rates
// returns: average squared error, percentage correct
func (net *Network) Train(data []*Datum, learningRate float64) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		if len(d.Outputs) != len(net.outputs) {
			return 0, 0, errors.Errorf("Can't train network, len(data[%d].Outputs) != len(net.outputs) (%d != %d)", i, len(d.Outputs), len(net.outputs))
		}

		outs, err := net.Correct(d.Inputs, d.Outputs, learningRate)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't train network, correction failed on datum %d", i)
		}

		// @CHANGE : should allow for multiple correct answer
		testRankings := sortValues(d.Outputs)
		rankings := sortValues(outs)
		if rankings[0] == testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}

		sqrdErr, err := SquaredError(outs, d.Outputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't train network, failed to get squared error on datum %d", i)
		}

		avgErr += sqrdErr / float64(len(data))
	}

	return avgErr, percentCorrect, nil
}

// returns: average squared error, percentage correct
func (net *Network) Test(data []*Datum) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		if len(d.Outputs) != len(net.outputs) {
			return 0, 0, errors.Errorf("Can't test network, len(data[%d].Outputs) != len(net.outputs) (%d != %d)", i, len(d.Outputs), len(net.outputs))
		}

		outs, err := net.GetOutputs(d.Inputs, true)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't test network, getting outputs of network with data[%d] failed", i)
		}

		// @CHANGE : should allow for multiple correct answers
		testRankings := sortValues(d.Outputs)
		rankings := sortValues(outs)
		if rankings[0] == testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}

		sqrdErr, err := SquaredError(outs, d.Outputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't test network, failed to get squared error with data[%d]", i)
		}

		avgErr += sqrdErr / float64(len(data))
	}

	return avgErr, percentCorrect, nil
}
