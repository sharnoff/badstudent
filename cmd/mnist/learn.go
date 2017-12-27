package main

import (
	smartlearn "github.com/sharnoff/smartlearning/smartlearning"
	"github.com/pkg/errors"
	"fmt"
	"math"
)

func (img *image) convertToDatum() (*smartlearn.Datum) {
	d := new(smartlearn.Datum)
	d.Inputs = make([]float64, len(img.data))
	for i, f := range img.data {
		d.Inputs[i] = math.Pow(float64(f) / 256, 2)
	}
	d.Outputs = make([]float64, len(img.label))
	for i := range img.label {
		d.Outputs[i] = float64(img.label[i])
	}
	return d
}

func convertToData(imgs []*image) ([]*smartlearn.Datum) {
	ds := make([]*smartlearn.Datum, len(imgs))
	for i := range imgs {
		ds[i] = imgs[i].convertToDatum()
	}
	return ds
}

func main() {
	fmt.Print("Initializing network...")
	net, err := smartlearn.New(imgSize, numOptions)
	if err != nil {
		fmt.Printf("%s\n", errors.Wrap(err, "couldn't create new network\n"))
		return
	}

	{
		l, err := net.Add("inputs", smartlearn.Blank(), []int{imgSize})
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		l, err = net.Add("hl", smartlearn.Neurons(), []int{500}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		l, err = net.Add("hl-tanh", smartlearn.Tanh(), []int{500}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		l, err = net.Add("outputs", smartlearn.Neurons(), []int{numOptions}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		l, err = net.Add("outputs-tanh", smartlearn.Tanh(), []int{numOptions}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		l, err = net.Add("softmax", smartlearn.Softmax(), []int{numOptions}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network\n"))
			return
		}

		err = net.SetOutputs(l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't set outputs to network\n"))
			return
		}
	}
	fmt.Println(" Done!")

	// create the datasets that the network will be using
	var trainData, testData []*smartlearn.Datum
	{
		fmt.Print("Fetching training images...")
		trainImgs, err := getImgsFromFile("resources/mnist_train.csv")
		if err != nil {
			fmt.Printf("\n%s\n\t- in mnist.main()\n", err.Error())
			return
		}
		fmt.Println(" Done!")

		fmt.Print("Fetching testing images...")
		testImgs, err := getImgsFromFile("resources/mnist_test.csv")
		if err != nil {
			fmt.Printf("\n%s\n\t- in mnist.main()\n", err.Error())
			return
		}
		fmt.Println(" Done!\n")
		fmt.Print("Converting training images to data...")
		trainData = convertToData(trainImgs)
		fmt.Println(" Done!")
		fmt.Print("Converting testing images to data...")
		testData = convertToData(testImgs)
		fmt.Println(" Done!\n")
	}

	learningRate := 0.01
	maxEons := 10
	var trainE, testE float64
	var trainP, testP float64

	// initial test:
	fmt.Print("Performing inital test...")
	testE, testP, err = net.Test(testData)
	if err != nil {
		fmt.Printf("\n%s\n", errors.Wrapf(err, "error in initial testing of network\n"))
		return
	}
	fmt.Printf("Done! Test error: %g, Test %% correct: %g\n", testE, testP)

	fmt.Printf("Starting training for %d eons...\n", maxEons)
	fmt.Println("Format is: Eon # : training error, training % correct, testing error, testing % correct")
	for eon := 1; eon <= maxEons; eon++ {
		trainE, trainP, err = net.Train(trainData, learningRate)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrapf(err, "error in training during eon %d\n", eon))
			return
		}
		testE, testP, err = net.Test(testData)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrapf(err, "error in testing during eon %d\n", eon))
			return
		}
		fmt.Printf("%d : %g, %g, %g, %g\n", eon, trainE, trainP, testE, testP)
	}

	fmt.Println("Running final tests...")
	numTests := 100
	for i := 0; i < numTests; i++ {
		outs, err := net.GetOutputs(testData[i].Inputs)
		if err != nil {
			fmt.Printf("%s\n\t- in mnist.main() at test %d", err.Error(), i)
		}
		fmt.Printf("%v → %v\n", testData[i].Outputs, outs)
	}

	fmt.Println("Done testing, exiting program...")
	// ideally I would have a way of saving the network here...
	return
}
