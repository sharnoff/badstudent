package main

import (
	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/optimizers"

	"github.com/pkg/errors"
	"fmt"

	"os"
	"bufio"
	"strconv"
	"strings"
)

const (
	imgSize    int = 784 // 28x28
	numOptions int = 10  // 0->9
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
	for i := 0;; i++ {
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
				req := <- request
				
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

// // assumes that it take the output of trainData() as args
// func trainCh(data [][][]uint8, err error) (func(int, int, chan badstudent.AnnotatedDatum, *error), error) {

// 	if err != nil {
// 		return nil, err
// 	}

// 	return func(amount, start int, ch chan badstudent.AnnotatedDatum, err *error) {

// 		for i := 0; i < amount; i++ {
// 			index := (start + i) % len(data)

// 			var epoch bool
// 			if (index + 1) % 1000 == 0 {
// 				epoch = true
// 			}

// 			// if index == len(data) - 1 {
// 			// 	epoch = true
// 			// }

// 			d := data[index]
// 			datum := badstudent.Datum{make([]float64, len(d[0])), make([]float64, len(d[1]))}
// 			for in := range d[0] {
// 				datum.Inputs[in] = float64(d[0][in])
// 			}

// 			for out := range d[1] {
// 				datum.Outputs[out] = float64(d[1][out])
// 			}

// 			ch <- badstudent.AnnotatedDatum{datum, epoch}
// 		}

// 		close(ch)
// 		return
// 	}, nil
// }

// func testData(fileName string) (func(int, int, chan badstudent.AnnotatedDatum, *error), error) {
// 	// attempt to open, then close the file to make sure it's there
// 	{
// 		f, err := os.Open(fileName)
// 		if err != nil {
// 			return nil, errors.Wrapf(err, "Couldn't open file %s\n", fileName)
// 		}

// 		f.Close()
// 	}

// 	return func(amount, start int, ch chan badstudent.AnnotatedDatum, errPtr *error) {
// 		f, err := os.Open(fileName)
// 		if err != nil {
// 			*errPtr = errors.Wrapf(err, "Couldn't open file %s\n", fileName)
// 			close(ch)
// 			return
// 		}

// 		defer close(ch)
// 		defer f.Close()

// 		sc := bufio.NewScanner(f)
// 		for i := 0;; i++ {
// 			if !sc.Scan() {
// 				break
// 			}

// 			s := sc.Text()

// 			img, err := image(s)
// 			if err != nil {
// 				*errPtr = errors.Wrapf(err, "Coudln't get image from line (%d in test)\n", i)
// 				return
// 			}

// 			d := badstudent.Datum{make([]float64, len(img[0])), make([]float64, len(img[1]))}
// 			for i := range img[0] {
// 				d.Inputs[i] = float64(img[0][i])
// 			}
// 			for i := range img[1] {
// 				d.Outputs[i] = float64(img[1][i])
// 			}

// 			ch <- badstudent.AnnotatedDatum{d, false}
// 		}

// 		return
// 	}, nil
// }

// because it's only run with 4 args, we can make certain shortcuts
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
	maxIterations := 600000 // 600 000
	batchSize := 10
	testFrequency := 2000
	statusFrequency := 1000
	trainFile := "resources/mnist_train.csv"
	testFile := "resources/mnist_test.csv"

	fmt.Println("Initializing network...")
	net := new(badstudent.Network)
	{
		l, err := net.Add("inputs", operators.Neurons(), imgSize, nil, nil)
		if err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("hidden layer neurons", operators.Neurons(), 500, nil, optimizers.GradientDescent(), l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("hidden layer logistic", operators.Logistic(), 500, nil, nil, l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output neurons", operators.Neurons(), 10, nil, optimizers.GradientDescent(), l); err != nil {
			panic(err.Error())
		}

		if l, err = net.Add("output logistic", operators.Logistic(), 10, nil, nil, l); err != nil {
			panic(err.Error())
		}

		if err = net.SetOutputs(l); err != nil {
			panic(err.Error())
		}
	}
	fmt.Println("Done!")

	fmt.Println("Fetching data...")
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
	fmt.Println("Done!")
	
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

	fmt.Println("Starting training for", maxIterations, "iterations!")
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

	fmt.Println("Done training!")
}
