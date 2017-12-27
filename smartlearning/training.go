package smartlearning

import (
	"github.com/pkg/errors"
	"sort"
)

func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if len(inputs) != len(net.inputs) {
		return nil, errors.Errorf("couldn't get outputs: len(inputs) != len(net.inputs) (%d != %d)", len(inputs), len(net.inputs))
	}

	ind := 0
	for i, in := range net.inSegments {
		end := ind + len(in.values)
		if end > len(net.inputs) {
			return nil, errors.Errorf("arcane error in (*Network).GetOutputs(): end > len(net.inputs) for net input %d (name: %s) (%d > %d)", i, in.name, end, len(net.inputs))
		}
		err := in.SetValues(inputs[ind:end])
		if err != nil {
			return nil, errors.Wrapf(err, "couldn't set values of net input %d (name: %s)\n", i, in.name)
		}
		ind = end
	}

	results := make([]chan error, len(net.outSegments))
	for i, o := range net.outSegments {
		results[i] = make(chan error)
		go o.Calculate(results[i])
	}
	for i, ch := range results {
		err := <-ch
		if err != nil {
			return nil, errors.Wrapf(err, "couldn't calculate net output %d (name: %s)\n", i, net.outSegments[i].name)
		}
	}

	return net.outputs, nil
}

func (net *Network) AdjustToTargets(targets []float64, learningRate float64) error {
	if len(net.outputs) != len(targets) {
		return errors.Errorf("can't adjust to targets : len(net.outputs) != len(targets) (%d != %d)", len(net.outputs), len(targets))
	}

	err := net.GenerateDeltas(targets)
	if err != nil {
		return errors.Wrap(err, "could not generate deltas\n")
	}

	results := make([]chan error, len(net.outSegments))
	for i, o := range net.outSegments {
		results[i] = make(chan error)
		go o.Adjust(learningRate, true, results[i]) // 'true' means that it will recurse downwards
	}
	for i, ch := range results {
		err := <-ch
		if err != nil {
			return errors.Wrapf(err, "coudn't adjust net output %d (name: %s)\n", i, net.outSegments[i].name)
		}
	}

	return nil
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

func sortValues(vs []float64, duplicate bool) []int {
	if duplicate {
		c := make([]float64, len(vs))
		copy(c, vs)
		vs = c
	}
	arr := arrWithIndexes{vs, make([]int, len(vs))}
	for i := range arr.indexes {
		arr.indexes[i] = i
	}
	sort.Sort(arr)
	return arr.indexes
}

type Datum struct {
	Inputs, Outputs []float64
}

// @CHANGE : allow for changing learning rates
// @CHANGE : allow for different error functions
// returns: average squared error, percentage correct, error
func (net *Network) Train(data []*Datum, learningRate float64) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		// doesn't need to check inputs bc GetOutputs() does that
		if len(d.Outputs) != len(net.outputs) {
			return 0, 0, errors.Errorf("len(d.Outputs) != len(net.outputs) (%d != %d) - at datum %d", len(d.Outputs), len(net.outputs), i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "tried to get the outputs of the network at datum %d\n", i)
		}

		// @CHANGE : should allow for multiple correct answers (see net.Test())
		rankings := sortValues(d.Outputs, true)
		// this following line could be more efficient, but it's needed for later features
		testRankings := sortValues(d.Outputs, true)
		// if the answer was correct
		if rankings[0] == testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}
		// add to the average squared error
		sqrdErr, _ := SquaredError(outs, d.Outputs)
		// this function won't return an error, because we already know that they're the same length
		avgErr += sqrdErr / float64(len(data))

		// correct the network
		err = net.AdjustToTargets(d.Outputs, learningRate)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "tried to adjust network with datum %d\n", i)
		}
	}

	return avgErr, percentCorrect, nil
}

// returns: average squared error, percentage correct, error
// only works with 0s and 1s in outputs for data
func (net *Network) Test(data []*Datum) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		// doesn't need to check inputs bc GetOutputs() does that
		if len(d.Outputs) != len(net.outputs) {
			return 0, 0, errors.Errorf("len(d.Outputs) != len(net.outputs) (%d != %d) - at datum %d", len(d.Outputs), len(net.outputs), i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "tried to get the outputs of the network at datum %d\n", i)
		}

		// @CHANGE : should allow for multiple correct answers (see net.Train())
		rankings := sortValues(outs, true)
		testRankings := sortValues(d.Outputs, true)
		if rankings[0] == testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}
		sqrdErr, _ := SquaredError(outs, d.Outputs)
		avgErr += sqrdErr / float64(len(data))
	}

	return avgErr, percentCorrect, nil
}
