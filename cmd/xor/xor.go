package main

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/operators/optimizers"

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

const (
	statusFrequency int = 100
	testFrequency   int = 1000

	// main hyperparameters
	learningRate  float64 = 0.2
	batchSize     int     = 4
	maxIterations int     = 100000

	// where to save/load the network
	path string = "xor save"
)

func train(net *bs.Network, dataset [][][]float64) {
	trainData, err := bs.Data(dataset, bs.BatchEvery(batchSize))
	if err != nil {
		panic(err.Error())
	}

	testData, err := bs.Data(dataset, bs.BatchEvery(len(dataset)))
	if err != nil {
		panic(err.Error())
	}

	res := make(chan bs.Result)

	args := bs.TrainArgs{
		Data:         trainData,
		TestData:     testData,
		ShouldTest:   bs.Every(testFrequency),
		SendStatus:   bs.Every(statusFrequency),
		RunCondition: bs.TrainUntil(maxIterations),
		LearningRate: bs.ConstantRate(learningRate),
		// IsCorrect: bs.CorrectRound,
		// CostFunc:  bs.SquaredError(false),
		Results: res,
		Err:     &err,
	}

	fmt.Println("Starting training...")
	go net.Train(args)
	// fmt.Println("Iteration, Status Cost, Status Percent Correct, Test Cost, Test Percent Correct")

	// statusCost, statusPercent, testCost, testPercent
	results := make([]float64, 4)
	previousIteration := -1

	for r := range res {

		if r.Iteration > previousIteration && previousIteration >= 0 {
			// fmt.Printf("%d, %s\n", previousIteration, format(results...))

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
	}

	if err != nil {
		panic(err.Error())
	}

	// fmt.Printf("%d, %s\n", previousIteration, format(results...))
	fmt.Println("Done training!")
}

func test(net *bs.Network, dataset [][][]float64) {
	testData, err := bs.Data(dataset, bs.BatchEvery(len(dataset)))
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Testing...")
	_, _, err = net.Test(testData, bs.SquaredError(true), bs.CorrectRound)
	if err != nil {
		panic(err.Error())
	}
}

func save(net *bs.Network) {
	fmt.Println("Saving...")
	path := "xor save"
	if err := net.Save(path, true); err != nil {
		panic(err.Error())
	}
	fmt.Println("Done!")
}

func load() (net *bs.Network) {

	fmt.Println("Loading...")
	path := "xor save"
	types := map[string]bs.Operator{
		"input":                 operators.Neurons(optimizers.GradientDescent()),
		"hidden layer neurons":  operators.Neurons(optimizers.GradientDescent()),
		"loop":                  operators.Neurons(optimizers.GradientDescent()),
		"loop logistic":         operators.Logistic(),
		"hidden layer logistic": operators.Logistic(),
		"output neurons":        operators.Neurons(optimizers.GradientDescent()),
		"output logistic":       operators.Logistic(),
	}
	// no auxiliary information necessary
	var err error
	if net, err = bs.Load(path, types, nil); err != nil {
		panic(err.Error())
	}
	fmt.Println("Done!")

	return
}

func main() {
	dataset := [][][]float64{
		{{-1, -1}, {0}},
		{{-1, 1}, {1}},
		{{1, -1}, {1}},
		{{1, 1}, {0}},
	}

	net := new(bs.Network)
	fmt.Println("Setting up network...")
	{
		var err error
		var l, loop, loopAF *bs.Node

		if l, err = net.Add("input", operators.Neurons(optimizers.GradientDescent()), 2, 0); err != nil {
			panic(err.Error())
		}

		if loop, err = net.Placeholder("loop", 1); err != nil {
			panic(err.Error())
		}

		if loopAF, err = net.Add("loop logistic", operators.Logistic(), 1, 0, loop); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("hidden layer neurons", operators.Neurons(optimizers.GradientDescent()), 2, 0, l, loopAF); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("hidden layer logistic", operators.Logistic(), 2, 0, l); err != nil {
			panic(err.Error())
		}

		if err = loop.Replace(operators.Neurons(optimizers.GradientDescent()), 1, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output neurons", operators.Neurons(optimizers.GradientDescent()), 1, 0, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output logistic", operators.Logistic(), 1, 0, l); err != nil {
			panic(err.Error())
		}

		if err = net.SetOutputs(l); err != nil {
			panic(err.Error())
		}
	}
	fmt.Println("Done!")

	train(net, dataset)
	test(net, dataset)
	save(net)
	net = load()
	train(net, dataset)
	test(net, dataset)
	save(net)

	// net := load()
	// train(net, dataset)
	// test(net, dataset)
	// save(net)
}
