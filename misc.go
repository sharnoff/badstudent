package badstudent

import (
	"math"
	"sort"
)

// CorrectRound is the default 'IsCorrect' function to be provided to TrainArgs
//
// A value is correct if it is on the same side of 0.5 as the target
// -- values less than 0.5 round to 0, and values greater than or equal to 0.5 round to 1
// This requires 'targets' to consist of either '0' or '1'
func CorrectRound(outs, targets []float64) bool {
	for i := range outs {
		// rounds to 0 if a number is < 0.5, 1 if ≥ 0.5. Tanh reduces the value to (0, 1)
		if math.Round(0.5*(1+math.Tanh(outs[i]-0.5))) != targets[i] {
			return false
		}
	}

	return true
}

// An alternate 'IsCorrect' function to provide to TrainArgs
func CorrectHighest(outs, targets []float64) bool {
	return HighestIndex(outs) == HighestIndex(targets)
}

// for use in HighestIndexes
type sortable struct {
	values  []float64
	indexes []int
}

func (s sortable) Len() int {
	return len(s.values)
}
func (s sortable) Less(i, j int) bool {
	return s.values[i] > s.values[j]
}
func (s sortable) Swap(i, j int) {
	s.values[i], s.values[j] = s.values[j], s.values[i]
	s.indexes[i], s.indexes[j] = s.indexes[j], s.indexes[i]
	return
}

// returns the indexes of the highest values in the given slice, from greatest to least
func HighestIndexes(sl []float64) []int {
	indexes := make([]int, len(sl))
	for i := range indexes {
		indexes[i] = i
	}

	s := sortable{sl, indexes}
	sort.Sort(s)

	return s.indexes
}

// returns the index of the highest value in an unsorted slice
func HighestIndex(sl []float64) int {
	highVal := math.Inf(-1)
	index := -1
	for i, v := range sl {
		if v > highVal {
			highVal = v
			index = i
		}
	}

	return index
}

// Acts as a TrainArgs.RunCondition
// Tells the network to run for a specified number of individual data corrected
func TrainUntil(maxIterations int) func(int, float64) bool {
	return func(iteration int, lastErr float64) bool {
		return iteration < maxIterations
	}
}

// Acts as a TrainArgs.LearningRate
// Always returns the provided 'learningRate'
func ConstantRate(learningRate float64) func(int, float64) float64 {
	return func(iteration int, lastErr float64) float64 {
		return learningRate
	}
}

// Returns a function that satisfies multiple fields in TrainArgs
//
// returns (iteration % frequency == 0)
func Every(frequency int) func(int) bool {
	return func(iteration int) bool {
		return iteration%frequency == 0
	}
}

// returns (iteration % frequency == frequency - 1)
//
// Works for the TrainArgs fields that should return true only at the end
// of a range
func EndEvery(frequency int) func(int) bool {
	return func(iteration int) bool {
		return iteration%frequency == frequency-1
	}
}
