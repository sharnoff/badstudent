package badstudent

import (
	"math"
	"sort"
)

// assumes len(outs) == len(targets)
func CorrectRound(outs, targets []float64) bool {
	for i := range outs {
		// rounds to 0 if a number is < 0.5, 1 if > 0.5. Tanh reduces the value to (0, 1)
		if math.Round(0.5 * (1 + math.Tanh(outs[i] - 0.5))) != targets[i] {
			return false
		}
	}

	return true
}

// for use in CorrectHighest()
type sortable struct {
	values []float64
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

// just returns whether or not the largest value in each is the same
func CorrectHighest(outs, targets []float64) bool {
	indexes := make([]int, len(outs))
	for i := range indexes {
		indexes[i] = i
	}

	copyOfIndexes := make([]int, len(outs))
	copy(copyOfIndexes, indexes)

	o := sortable{outs, indexes}
	t := sortable{targets, copyOfIndexes}

	sort.Sort(o)
	sort.Sort(t)

	return o.indexes[0] == t.indexes[0]
}

func TrainUntil(maxIterations int) func(int, float64) bool {
	return func(iteration int, lastErr float64) bool {
		return iteration < maxIterations
	}
}

// returns a function that satisfies TrainArgs.LearningRate
func ConstantRate(learningRate float64) func(int, float64) float64 {
	return func(iteration int, lastErr float64) float64 {
		return learningRate
	}
}

// returns a function that satisfies TrainArgs.SendStatus
// 'frequency' is in units of iterations
//
// this function is self-explanatory from viewing the source
func Every(frequency int) func(int) bool {
	return func(iteration int) bool {
		return iteration % frequency == 0
	}
}

// returns a function that satisfies TrainArgs.Batch
// 'frequency' is in units of iterations
//
// this function is self-explanatory from viewing the source
func BatchEvery(frequency int) func(int) (bool, bool) {
	if frequency == 1 {
		return func(iteration int) (bool, bool) {
			return true, false
		}
	} else {
		return func(iteration int) (bool, bool) {
			return (iteration % frequency == 0), true
		}
	}
}

// returns a function that satisfies TrainArgs.ShouldTest
// 'frequency' is in units of iterations
// 'amount' is the quantity of test data that should be tested on
// 
// this function is self-explanatory from viewing the source
func TestEvery(frequency, amount int) func(int) int {
	return func(iteration int) int {
		if iteration % frequency == 0 {
			return amount
		}

		return 0
	}
}
