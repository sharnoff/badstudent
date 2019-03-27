package badstudent

import (
	"fmt"
	"math"
	"sort"
)

func format(fs ...float64) (str string) {
	for i := range fs {
		if fs[i] != 0 {
			str += fmt.Sprintf("%v", fs[i])
		}
		str += ", "
	}

	return
}

// PrintResult returns a function that prints the results of training, in addition to a final
// function to be called once the training has finished.
//
// It prints: Iteration, Status Cost, Status Percent, Test Cost, Test Percent
//
// The final results will not be printed unless the secondary function is called.
func PrintResult() (func(Result), func()) {
	// statusCost, statusPercent, testCost, testPercent
	results := make([]float64, 4)
	previousIteration := -1

	return func(r Result) {
			if r.Iteration > previousIteration && previousIteration >= 0 {
				fmt.Printf("%d, %s\n", previousIteration, format(results...))

				results = make([]float64, len(results))
			}

			if r.IsTest {
				results[2] = r.Cost
				results[3] = r.Correct * 100
			} else {
				results[0] = r.Cost
				results[1] = r.Correct * 100
			}

			previousIteration = r.Iteration
		},
		func() {
			fmt.Printf("%d, %s\n", previousIteration, format(results...))
		}
}

// CorrectRound is the default 'IsCorrect' function to be provided to TrainArgs
//
// A value is correct if it is on the same side of 0.5 as the target -- values less than 0.5 round
// to 0, and values greater than or equal to 0.5 round to 1 This requires 'targets' to consist of
// either '0' or '1'
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
func TrainUntil(maxIterations int) func(int) bool {
	return func(iteration int) bool {
		return iteration < maxIterations
	}
}

// Every returns a function that returns (iteration % frequency == 0)
func Every(frequency int) func(int) bool {
	return func(iteration int) bool {
		return iteration%frequency == 0
	}
}

// EndEvery returns a function that returns (iteration % frequency == frequency - 1)
func EndEvery(frequency int) func(int) bool {
	return func(iteration int) bool {
		return iteration%frequency == frequency-1
	}
}
