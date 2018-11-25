package main

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/costfuncs"
	"github.com/sharnoff/badstudent/hyperparams"
	"github.com/sharnoff/badstudent/initializers"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/optimizers"
	"github.com/sharnoff/badstudent/penalties"

	"fmt"
	"time"
)

const (
	statusFrequency int = 100
	testFrequency   int = 1000

	// main hyperparameters
	learningRate  float64 = 0.1
	batchSize     int     = 1
	maxIterations int     = 10000

	// where to save/load the network
	path string = "xor save"
)

func train(net *bs.Network, dataset [][][]float64) {
	trainData, err := bs.SeqData(dataset, batchSize, len(dataset))
	if err != nil {
		panic(err.Error())
	}

	testData, err := bs.SeqData(dataset, batchSize, len(dataset))
	if err != nil {
		panic(err.Error())
	}

	up, final := bs.PrintResult()

	args := bs.TrainArgs{
		TrainData:    trainData,
		TestData:     testData,
		ShouldTest:   bs.EndEvery(testFrequency),
		SendStatus:   bs.Every(statusFrequency),
		RunCondition: bs.TrainUntil(maxIterations),
		// IsCorrect: bs.CorrectRound,
		Update: up,
	}

	fmt.Println("Starting training...")
	startTime := time.Now()

	// fmt.Println("Iteration, Status Cost, Status Percent Correct, Test Cost, Test Percent Correct")
	if err := net.Train(args); err != nil {
		panic(err.Error())
	}

	final()
	fmt.Println("Done training! It took", time.Since(startTime).Seconds(), "seconds")
}

func test(net *bs.Network, dataset [][][]float64) {
	testData, err := bs.SeqData(dataset, len(dataset), len(dataset))
	if err != nil {
		panic(err.Error())
	}

	fmt.Println("Testing...")
	_, _, err = net.Test(testData, bs.CorrectRound)
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
	var err error
	if net, err = bs.Load(path); err != nil {
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
	/*{
		l := net.Add("input", nil, 2)
		l = net.Add("hidden neurons", operators.Neurons(), 3, l).Opt(optimizers.SGD())
		l = net.Add("hidden logistic", operators.Logistic(), 3, l)
		l = net.Add("output neurons", operators.Neurons(), 1, l).Opt(optimizers.SGD())
		l = net.Add("output logistic", operators.Logistic(), 1, l)

		net.AddHP("learning-rate", hyperparams.Constant(learningRate))
		if err := net.Finalize(costfuncs.MSE(), l); err != nil {
			panic(err.Error())
		}
	}*/
	{
		l := net.Add("input", nil, 2)
		loop := net.Placeholder("loop", 1)
		loopAF := net.Add("loop logistic", operators.Logistic(), 1, loop)

		l = net.Add("hidden layer neurons", operators.Neurons(), 2, l, loopAF).Opt(optimizers.SGD())
		l = net.Add("hidden layer logistic", operators.Logistic(), 2, l)

		loop.Replace(operators.Neurons(), l).SetDelay(1).Opt(optimizers.SGD()).SetPenalty(penalties.ElasticNet(0.5, 0.01))

		l = net.Add("output neurons", operators.Neurons(), 1, l).Opt(optimizers.SGD()).SetPenalty(penalties.ElasticNet(0.5, 0.001))
		l = net.Add("output logistic", operators.Logistic(), 1, l)

		net.DefaultInit(initializers.Xavier())
		net.AddHP("learning-rate", hyperparams.Step(learningRate).Add(5000, learningRate / 10))
		if err := net.Finalize(costfuncs.MSE(), l); err != nil {
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

	/*
	if err := net.Graph("graph"); err != nil {
		panic(err.Error())
	}
	// */

	// net = load()
	// train(net, dataset)
	// test(net, dataset)
	// save(net)
}
