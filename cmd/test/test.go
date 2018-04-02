package main

import (
	"github.com/sharnoff/smartlearning/smartlearn"

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
	learningRate := 0.1
	maxEpochs := 2000

	fmt.Println("Setting up network...")
	net := new(smartlearn.Network)
	{
		var err error

		if err = net.Add("inputs", 2); err != nil {
			panic(err.Error())
		}

		if err = net.Add("hidden layer neurons", 3); err != nil {
			panic(err.Error())
		}

		if err = net.Add("output neurons", 1); err != nil {
			panic(err.Error())
		}

		if err = net.SetOutputs(); err != nil {
			panic(err.Error())
		}
	}
	fmt.Println("Done!")

	res := make(chan struct {
		Avg, Percent  float64
		Epoch, IsTest bool
	})

	dataSrc, err := smartlearn.TrainCh(dataset)
	if err != nil {
		panic(err.Error())
	}

	args := smartlearn.TrainArgs{
		Data:         dataSrc,
		Results:      res,
		Err:          &err,
	}

	fmt.Println("Starting training...")
	go net.Train(args, maxEpochs, learningRate)

	for r := range res {
		if r.Epoch {
			// fmt.Printf("Train → avg error: %v\t → percent correct: %v from EPOCH\n", r.Avg, r.Percent)
		} else { // it should never be a test, because we didn't give it testing data
			// fmt.Printf("Train → avg error: %v\t → percent correct: %v\n", r.Avg, r.Percent)
		}
	}

	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Done training!")
}
