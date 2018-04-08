package badstudent

import (
	"math"
)

// assumes len(outs) == len(targets_)
func CorrectRound(outs, targets []float64) bool {
	for i := range outs {
		// rounds to 0 if a number is < 0.5, 1 if > 0.5. Tanh reduces the value to (0, 1)
		if math.Round(0.5 * (1 + math.Tanh(outs[i] - 0.5))) != targets[i] {
			return false
		}
	}

	return true
}

// returns a function that satisfies TrainArgs.RunCondition
func TrainFor(maxEpochs int) func(int, int, float64) bool {
	return func(iteration, epoch int, lastErr float64) bool {
		return epoch < maxEpochs
	}
}

// returns a function that satisfies TrainArgs.LearningRate
func ConstantRate(learningRate float64) func(int, int, float64) float64 {
	return func(iteration, epoch int, lastErr float64) float64 {
		return learningRate
	}
}
