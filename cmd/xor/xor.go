package main

import (
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/optimizers"

	"fmt"
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

func main() {
	dataset := [][][]float64{
		{{-1, -1}, {0}},
		{{-1, 1}, {1}},
		{{1, -1}, {1}},
		{{1, 1}, {0}},
	}

	// these are the main adjustable variables
	statusFrequency := 100
	testFrequency := 20

	// primary set of hyperparameters
	learningRate := 1.0
	batchSize := 4
	maxIterations := 3000 // 750 epochs

	// alternate set that works
	// learningRate := 1.0
	// batchSize := 1
	// maxIterations := 1200

	fmt.Println("Setting up network...")
	net := new(badstudent.Network)
	{
		var err error
		var l, hl *badstudent.Layer

		if l, err = net.Add("input", operators.Neurons(), 2, nil, nil); err != nil {
			panic(err.Error())
		}

		if hl, err = net.Add("hidden layer neurons", operators.Neurons(), 1, nil, optimizers.GradientDescent(), l); err != nil {
			panic(err.Error())
		}

		if hl, err = net.Add("hidden layer logistic", operators.Logistic(), 1, nil, nil, hl); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output neurons", operators.Neurons(), 1, nil, optimizers.GradientDescent(), hl, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output logistic", operators.Logistic(), 1, nil, nil, l); err != nil {
			panic(err.Error())
		}

		if err = net.SetOutputs(l); err != nil {
			panic(err.Error())
		}
	}
	fmt.Println("Done!")

	trainData, err := badstudent.DataCh(dataset, true)
	if err != nil {
		panic(err.Error())
	}

	testData, err := badstudent.DataCh(dataset, false)
	if err != nil {
		panic(err.Error())
	}

	res := make(chan badstudent.Result)

	args := badstudent.TrainArgs{
		Data:         trainData,
		TestData:     testData,
		ShouldTest:   badstudent.TestEvery(testFrequency, len(dataset)),
		SendStatus:   badstudent.Every(statusFrequency),
		Batch:        badstudent.BatchEvery(batchSize),
		RunCondition: badstudent.TrainUntil(maxIterations),
		LearningRate: badstudent.ConstantRate(learningRate),
		// IsCorrect: badstudent.CorrectRound,
		// CostFunc:  badstudent.SquaredError(false),
		Results:      res,
		Err:          &err,
	}

	fmt.Println("Starting training...")
	{
		go net.Train(args)
		fmt.Println("Iteration, Status Cost, Status Percent, Test Cost, Test Percent")
		
		// statusCost, statusPercent, testCost, testPercent
		results := make([]float64, 4)
		previousIteration := -1

		for r := range res {

			if r.Iteration > previousIteration && previousIteration >= 0 {
				fmt.Printf("%d, %s\n", previousIteration, format(results...))

				results = make([]float64, len(results))
			}

			if r.IsTest {
				results[2] = r.Avg
				results[3] = r.Percent
			} else {
				results[0] = r.Avg
				results[1] = r.Percent
			}

			previousIteration = r.Iteration
		}

		if err != nil {
			panic(err.Error())
		}

		fmt.Printf("%d, %s\n", previousIteration, format(results...))
	}
	fmt.Println("Done training!")

	fmt.Println("Testing...")
	{
		_, _, err = net.Test(testData, badstudent.SquaredError(true), badstudent.CorrectRound, len(dataset))
	}
}
