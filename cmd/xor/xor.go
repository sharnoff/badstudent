package main

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/costfuncs"
	"github.com/sharnoff/badstudent/hyperparams"
	_ "github.com/sharnoff/badstudent/initializers"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/optimizers"
	_ "github.com/sharnoff/badstudent/penalties"

	"fmt"
	"time"
)

const (
	statusFrequency int = 100
	testFrequency   int = 1000

	// main hyperparameters
	learningRate  float64 = 0.5
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
	if _, err := net.Save(path, true); err != nil {
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
	{
		l := net.AddInput([]int{2}).SetName("input")
		l = net.Add(operators.Neurons(3), l).Opt(optimizers.SGD()).SetName("hidden neurons")
		l = net.Add(operators.Logistic(), l).SetName("hidden logistic")
		l = net.Add(operators.Neurons(1), l).Opt(optimizers.SGD()).SetName("output neurons")
		l = net.Add(operators.Logistic(), l).SetName("output logistic")

		net.AddHP("learning-rate", hyperparams.Constant(learningRate))
		if err := net.Finalize(costfuncs.MSE(), l); err != nil {
			panic(err.Error())
		}
	}
	/*{
		l := net.AddInput([]int{2}).SetName("input")
		loop := net.Placeholder([]int{1}).SetName("loop")
		loopAF := net.Add(operators.Logistic(), loop).SetName("loop logistic")

		l = net.Add(operators.Neurons(2), l, loopAF).SetName("hidden layer neurons").Opt(optimizers.SGD())
		l = net.Add(operators.Logistic(), l).SetName("hidden layer logistic")

		loop.Replace(operators.Neurons(1), l).SetDelay(1).Opt(optimizers.SGD()).SetPenalty(penalties.ElasticNet(0.5, 0.01))

		l = net.Add(operators.Neurons(1), l).SetName("output neurons").Opt(optimizers.SGD()).SetPenalty(penalties.ElasticNet(0.5, 0.001))
		l = net.Add(operators.Logistic(), l).SetName("output logistic")

		net.DefaultInit(initializers.Xavier())
		net.AddHP("learning-rate", hyperparams.Step(learningRate).Add(5000, learningRate / 10))
		if err := net.Finalize(costfuncs.MSE(), l); err != nil {
			panic(err.Error())
		}
	}*/
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
