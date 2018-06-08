// requires resources/mnist_train.csv, resources/minst_test.csv to be in local path with format:
// <class>, img[0], img[1], img[2], ... img[783],
// <class>, img[0], img[1], img[2], ... img[783],
// ...
// where <class> is 0 -> 9 and img[n] is an integer in the range [0, 255]
//
// the constants below should be changed accordingly if anything other than the usual MNIST files are used

package main

import (
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/operators/optimizers"

	"fmt"
	"github.com/pkg/errors"

	"bufio"
	"os"
	"strconv"
	"strings"
)

const (
	imgSize    int = 784   // 28x28
	numOptions int = 10    // 0->9
	testSize   int = 10000 // 10 000
)

// classifier should be at index 0
func image(str string) ([][]uint8, error) {
	s := strings.Split(str, ",")

	if len(s) != (imgSize + 1) {
		return nil, errors.Errorf("Can't get image, not enough values from line (had %d, should be %d)", len(s), imgSize+1)
	}

	img := make([][]uint8, 2) // one for inputs, one for outputs

	{
		class, err := strconv.ParseInt(s[0], 10, 0)
		if err != nil {
			return nil, errors.Wrapf(err, "Couldn't parse value of classifier (given: %s)\n", s[0])
		} else if class >= int64(numOptions) {
			return nil, errors.Errorf("Classifier is out of bounds (%d >= %d)", class, numOptions)
		}

		img[1] = make([]uint8, numOptions)
		img[1][class] = 1
	}

	{
		img[0] = make([]uint8, imgSize)

		for i := 0; i < imgSize; i++ {
			v, err := strconv.ParseInt(s[i+1], 10, 0)
			if err != nil {
				return nil, errors.Wrapf(err, "Couldn't parse value %d of line (given: %s)\n", i, s[i])
			}

			img[0][i] = uint8(v)
		}
	}

	return img, nil
}

func data(fileName string) ([][][]uint8, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "Couldn't open file %s\n", fileName)
	}

	defer f.Close()

	var data [][][]uint8

	sc := bufio.NewScanner(f)
	for i := 0; ; i++ {
		if !sc.Scan() {
			break
		}

		str := sc.Text()
		img, err := image(str)
		if err != nil {
			return nil, errors.Wrapf(err, "Couldn't get image on line %d for file %s\n", i, fileName)
		}

		data = append(data, img)
	}

	if err = sc.Err(); err != nil {
		return nil, errors.Wrapf(err, "Scanning file %s encountered an error\n", fileName)
	}

	// trim the array to make storage smaller
	clone := make([][][]uint8, len(data))
	copy(clone, data)
	return clone, nil
}

func dataCh(data [][][]uint8, loop bool) func(chan bool, chan badstudent.Datum, *bool, *error) {

	return func(request chan bool, dataSrc chan badstudent.Datum, moreData *bool, err *error) {
		defer close(dataSrc)

		for {
			for _, d := range data {
				req := <-request

				if req == false {
					return
				}

				in := make([]float64, len(d[0]))
				for i := range in {
					in[i] = float64(d[0][i])
				}

				out := make([]float64, len(d[1]))
				for i := range out {
					out[i] = float64(d[1][i])
				}

				dataSrc <- badstudent.Datum{in, out}

			}

			if !loop {
				*moreData = false
				break
			}
		}

		return
	}
}

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
	learningRate := 0.001
	maxIterations := 10000 // 10 000
	batchSize := 100
	testFrequency := 2000
	statusFrequency := 1000
	trainFile := "resources/mnist_train.csv"
	testFile := "resources/mnist_test.csv"

	// fmt.Println("Initializing network...")
	net := new(badstudent.Network)
	{
		l, err := net.Add("inputs", operators.Neurons(optimizers.GradientDescent()), imgSize)
		if err != nil {
			panic(err.Error())
		}

		convArgs := operators.ConvArgs{
			Dims:        []int{28, 28},
			InputDims:   []int{28, 28},
			Filter:      []int{5, 5},
			ZeroPadding: []int{2, 2},
			Biases:      true,
		}
		if l, err = net.Add("conv-1", operators.Convolution(&convArgs, optimizers.GradientDescent()), 784, l); err != nil {
			panic(err.Error())
		}

		poolArgs := operators.PoolArgs{
			Dims:      []int{14, 14},
			InputDims: []int{28, 28},
			Filter:    []int{2, 2},
			Stride:    []int{2, 2},
		}
		if l, err = net.Add("pool-1", operators.AvgPool(&poolArgs), 196, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("pool-1 logistic", operators.Logistic(), 196, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output neurons", operators.Neurons(optimizers.GradientDescent()), 10, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output logistic", operators.Logistic(), 10, l); err != nil {
			panic(err.Error())
		}

		if err = net.SetOutputs(l); err != nil {
			panic(err.Error())
		}
	}
	// fmt.Println("Done!")

	// fmt.Println("Fetching data...")
	var trainSrc, testSrc func(chan bool, chan badstudent.Datum, *bool, *error)
	{
		trainData, err := data(trainFile)
		if err != nil {
			panic(err.Error())
		}

		testData, err := data(testFile)
		if err != nil {
			panic(err.Error())
		}

		trainSrc = dataCh(trainData, true)
		testSrc = dataCh(testData, false)
	}
	// fmt.Println("Done!")

	var res chan badstudent.Result = make(chan badstudent.Result)
	var err error

	args := badstudent.TrainArgs{
		Data:         trainSrc,
		TestData:     testSrc,
		ShouldTest:   badstudent.TestEvery(testFrequency, testSize),
		SendStatus:   badstudent.Every(statusFrequency),
		Batch:        badstudent.BatchEvery(batchSize),
		RunCondition: badstudent.TrainUntil(maxIterations),
		LearningRate: badstudent.ConstantRate(learningRate),
		IsCorrect:    badstudent.CorrectHighest,
		// CostFunc:  badstudent.SquaredError(false),
		Results: res,
		Err:     &err,
	}

	// fmt.Println("Starting training for", maxIterations, "iterations!")
	{
		go net.Train(args)

		fmt.Println("Iteration, Train Cost, Train %, Test Cost, Test %")

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
	// fmt.Println("Done training!")

	// fmt.Println("Saving...")
	{
		path := "mnist save"
		if err := net.Save(path, false); err != nil {
			panic(err.Error())
		}
	}
	// fmt.Println("Done!")
}
