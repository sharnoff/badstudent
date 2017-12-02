package smartlearning

import (
	errs "FromGithub/errors"
	"fmt"
	"sort"
)

func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if len(inputs) != net.numInputs {
		return nil, errs.Errorf("len(inputs) != net.numInputs (%d != %d)")
	}

	end := 0
	for i, in := range net.inputs {
		start := end
		end += len(in.values)
		if end > net.numInputs {
			return nil, errs.Errorf("ending for input %d is out of the bounds of the inputs (%d > %d)", i, end, net.numInputs)
		}
		err := in.SetValues(inputs[start:end])
		if err != nil {
			return nil, errs.Wrap(err, fmt.Sprintf("tried to set values of net input %d. starting index: %d, ending index: %d, number of inputs: %d", i, start, end, net.numInputs))
		}
	}

	setRange := func(base, sl []float64, start int) (int, error) {
		if start+len(sl) >= len(base) {
			return 0, errs.Errorf("start + len(sl) >= len(base) (%d + %d >= %d)", start, len(sl), len(base))
		}

		for i := range sl {
			base[start+i] = sl[i]
		}

		return start + len(sl), nil
	}

	// @OPTIMIZE: at setup, the outputs' values could all be subsets of the same slice
	outputs := make([]float64, net.numOutputs)
	index := 0
	for i, o := range net.outputs {

		outs, err := o.Values()
		if err != nil {
			return nil, errs.Wrap(err, fmt.Sprintf("tried to get values of output net output %d", i))
		}

		index, err = setRange(outputs, outs, index)
		if err != nil {
			return nil, errs.Wrap(err, "tried to set output values")
		}
	}

	return outputs, nil
}

func (net *Network) AdjustToTargets(targets []float64, learningRate float64) error {
	if net.numOutputs != len(targets) {
		return errs.Errorf("net.numOutputs != len(targets) (%d != %d)", net.numOutputs, len(targets))
	}

	for i, in := range net.inputs {
		err := in.CalcDeltas(net, targets)
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("tried to calculate delta of net input %d", i))
		}
	}

	for i, o := range net.outputs {
		err := o.Adjust(learningRate, true) // 'true' means that it will recurse downwards
		if err != nil {
			return errs.Wrap(err, fmt.Sprintf("tried to adjust net output %d with downward recursion", i))
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
// // not actually needed
// func (net *Network) SortOutupts() []int {
// 	return sortValues(net.outputs, true)
// }

type Datum struct {
	Inputs, Outputs []float64
}

// @CHANGE : allow for changing learning rates
// @CHANGE : allow for different error functions
// returns: average squared error, percentage correct, error
func (net *Network) Train(data []Datum, learningRate float64) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		// doesn't need to check inputs bc GetOutputs() does that
		if len(d.Outputs) != net.numOutputs {
			return 0, 0, errs.Errorf("len(d.Outputs) != net.numOutputs (%d != %d) - at datum %d", len(d.Outputs), net.numOutputs, i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errs.Wrap(err, fmt.Sprintf("tried to get the outputs of the network at datum %d", i))
		}

		// @CHANGE : should allow for multiple correct answers (see net.Test())
		rankings := sortValues(d.Outputs, true)
		// this following line could be more efficient, but it's needed for later features
		testRankings := sortValues(d.Outputs, true)
		// if the answer was correct
		if rankings[0] != testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}
		// add to the average squared error
		sqrdErr, _ := SquaredError(outs, d.Outputs)
		// this function won't return an error, because we already know that they're the same length
		avgErr += sqrdErr / float64(len(data))

		// correct the network
		err = net.AdjustToTargets(d.Outputs, learningRate)
		if err != nil {
			return 0, 0, errs.Wrap(err, fmt.Sprintf("tried to adjust network with datum %d", i))
		}
	}

	return avgErr, percentCorrect, nil
}

// returns: average squared error, percentage correct, error
// only works with 0s and 1s in outputs for data
func (net *Network) Test(data []Datum) (float64, float64, error) {
	percentCorrect, avgErr := 0.0, 0.0

	for i, d := range data {
		// doesn't need to check inputs bc GetOutputs() does that
		if len(d.Outputs) != net.numOutputs {
			return 0, 0, errs.Errorf("len(d.Outputs) != net.numOutputs (%d != %d) - at datum %d", len(d.Outputs), net.numOutputs, i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errs.Wrap(err, fmt.Sprintf("tried to get the outputs of the network at datum %d", i))
		}

		// @CHANGE : should allow for multiple correct answers (see net.Train() for more detailed comments)
		rankings := sortValues(outs, true)
		testRankings := sortValues(d.Outputs, true)
		if rankings[0] != testRankings[0] {
			percentCorrect += 100.0 / float64(len(data))
		}
		sqrdErr, _ := SquaredError(outs, d.Outputs)
		avgErr += sqrdErr / float64(len(data))
	}

	return avgErr, percentCorrect, nil
}
