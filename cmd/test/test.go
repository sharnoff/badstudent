package main

import (
	"github.com/sharnoff/smartlearning/badstudent"
	"github.com/sharnoff/smartlearning/badstudent/operators"
	"github.com/sharnoff/smartlearning/badstudent/optimizers"

	"fmt"
)

func main() {
	dataset := [][][]float64{
		{{-1, -1}, {0}},
		{{-1, 1}, {1}},
		{{1, -1}, {1}},
		{{1, 1}, {0}},
	}

	// these are the main adjustable variables
	learningRate := 1.0
	maxEpochs := 750
	batchSize := 4

	// alternate set that should work
	// learningRate := 1.0
	// maxEpochs := 300
	// batchSize := 1

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

	res := make(chan struct {
		Avg, Percent  float64
		Epoch, IsTest bool
	})

	dataSrc, err := badstudent.TrainCh(dataset)
	if err != nil {
		panic(err.Error())
	}

	testData, err := badstudent.TestCh(dataset)
	if err != nil {
		panic(err.Error())
	}

	args := badstudent.TrainArgs{
		Data:            dataSrc,
		TestData:        testData,
		TrainBeforeTest: 20,
		BatchSize:       batchSize,
		// IsCorrect:       badstudent.CorrectRound,
		// CostFunc:        badstudent.SquaredError(false),
		Results:         res,
		Err:             &err,
	}

	fmt.Println("Starting training...")
	go net.Train(args, maxEpochs, learningRate)

	for r := range res {
		if r.Epoch {
			fmt.Printf("Epoch → error: %v\t → %v%% correct from EPOCH\n", r.Avg, r.Percent)
		} else if r.IsTest {
			fmt.Printf("Test  → error: %v\t → %v%% correct\n", r.Avg, r.Percent)
		} else { // it should never be a test, because we didn't give it testing data
			// fmt.Printf("Train → error: %v\t → %v%% correct\n", r.Avg, r.Percent)
		}
	}

	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Done training!")
	fmt.Println("Testing!")

	testSrc, err := badstudent.TestCh(dataset)
	if err != nil {
		panic(err.Error())
	}

	_, _, err = net.Test(testSrc, badstudent.SquaredError(true), badstudent.CorrectRound)
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Done testing!")
}
