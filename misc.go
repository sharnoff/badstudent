package badstudent

import (
	"math"
)

// assumes len(outs) == len(targets)
func CorrectRound(outs, targets []float64) bool {
	for i := range outs {
		// rounds to 0 if a number is < 0.5, 1 if > 0.5. Tanh reduces the value to (0, 1)
		if math.Round(0.5*(1+math.Tanh(outs[i]-0.5))) != targets[i] {
			return false
		}
	}

	return true
}


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

// just returns whether or not the largest value in each is the same
func CorrectHighest(outs, targets []float64) bool {
	return HighestIndex(outs) == HighestIndex(targets)
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
		return iteration%frequency == 0
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
			return (iteration%frequency == 0), true
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
		if iteration%frequency == 0 {
			return amount
		}

		return 0
	}
}
