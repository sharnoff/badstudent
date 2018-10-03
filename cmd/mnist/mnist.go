// requires resources/mnist_train.csv, resources/minst_test.csv to be in local path with format:
// <class>, img[0], img[1], img[2], ... img[783],
// <class>, img[0], img[1], img[2], ... img[783],
// ...
// where <class> is 0 -> 9 and img[n] is an integer in the range [0, 255]
//
// the constants below should be changed accordingly if anything other than the usual MNIST files are used

package main

import (
	bs "github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/optimizers"

	"fmt"
	"github.com/pkg/errors"
	"time"

	"bufio"
	"os"
	"strconv"
	"strings"
)

const (
	imgSize    int     = 784 // 28x28
	numClasses int     = 10  // 0 -> 9
	maxInput   float64 = 255

	trainFile string = "resources/mnist_train.csv"
	testFile  string = "resources/mnist_test.csv"
	path      string = "resources/mnist_save"
)

type dataset struct {
	inputs  [][]uint8
	outputs []int // one-hot encoding = false

	index int
}

func (d *dataset) IsSequential() bool {
	return false
}

func (d *dataset) Get() (bs.Datum, error) {
	// cast inputs into []float64
	ins := make([]float64, imgSize)
	for i, in := range d.inputs[d.index] {
		ins[i] = float64(in) / maxInput
	}

	// transfer from one-hot encoding
	outs := make([]float64, numClasses)
	outs[d.outputs[d.index]] = 1

	d.index++
	if d.index >= len(d.inputs) {
		d.index = 0
	}

	return bs.Datum{ins, outs}, nil
}

func (d *dataset) SetEnded() bool {
	return true
}

func (d *dataset) BatchEnded() bool {
	return d.index%batchSize == 0
}

func (d *dataset) DoneTesting() bool {
	return d.index == 0
}

func image(str string) (ins []uint8, class int, err error) {
	s := strings.Split(str, ",")

	if len(s) != (imgSize + 1) {
		err = errors.Errorf("Can't get image, not enough values from line (had %d, should be %d)", len(s), imgSize+1)
		return
	}

	{
		var c int64
		c, err = strconv.ParseInt(s[0], 10, 0)
		class = int(c)
		if err != nil {
			err = errors.Wrapf(err, "Couldn't parse value of classifier (given: %s)\n", s[0])
			return
		} else if class >= numClasses {
			err = errors.Errorf("Classifier is out of bounds (%d >= %d)", class, numClasses)
			return
		}
	}

	{
		ins = make([]uint8, imgSize)

		for i := 0; i < imgSize; i++ {
			var v int64
			v, err = strconv.ParseInt(s[i+1], 10, 0)
			if err != nil {
				err = errors.Wrapf(err, "Couldn't parse value %d of line (given: %s)\n", i, s[i])
				return
			}

			ins[i] = uint8(v)
		}
	}

	return
}

func data(fileName string) (*dataset, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "Couldn't open file %s\n", fileName)
	}

	defer f.Close()

	ds := new(dataset)

	sc := bufio.NewScanner(f)
	for i := 0; ; i++ {
		if !sc.Scan() {
			break
		}

		str := sc.Text()
		ins, class, err := image(str)
		if err != nil {
			return nil, errors.Wrapf(err, "Couldn't get image on line %d for file %s\n", i, fileName)
		}

		ds.inputs = append(ds.inputs, ins)
		ds.outputs = append(ds.outputs, class)
	}

	if err = sc.Err(); err != nil {
		return nil, errors.Wrapf(err, "Scanning file %s encountered an error\n", fileName)
	}

	// trim the array to make storage smaller
	{
		inClone := make([][]uint8, len(ds.inputs))
		outClone := make([]int, len(ds.outputs))
		copy(inClone, ds.inputs)
		copy(outClone, ds.outputs)
		ds.inputs = inClone
		ds.outputs = outClone
	}

	return ds, nil
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

const (
	learningRate    float64 = 0.001
	maxIterations   int     = 10000 // 10 000
	batchSize       int     = 100
	testFrequency   int     = 20000
	statusFrequency int     = 1000
)

func train(net *bs.Network, trainData, testData *dataset) {
	res := make(chan bs.Result)
	var err error

	args := bs.TrainArgs{
		TrainData:    trainData,
		TestData:     testData,
		ShouldTest:   bs.EndEvery(testFrequency),
		SendStatus:   bs.Every(statusFrequency),
		RunCondition: bs.TrainUntil(maxIterations),
		LearningRate: bs.ConstantRate(learningRate),
		IsCorrect:    bs.CorrectHighest,
		Results:      res,
		Err:          &err,
	}

	fmt.Println("Starting training...")
	startTime := time.Now()
	go net.Train(args)
	fmt.Println("Iteration, Status Cost, Status Percent Correct, Test Cost, Test Percent Correct")

	// statusCost, statusPercent, testCost, testPercent
	results := make([]float64, 4)
	previousIteration := -1

	for r := range res {
		if r.Iteration > previousIteration && previousIteration >= 0 {
			fmt.Printf("%d, %s\n", previousIteration, format(results...))

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

	fmt.Printf("%d, %s\n", previousIteration, format(results...))
	fmt.Println("Done training! It took", time.Since(startTime).Seconds(), "seconds")
}

func test(net *bs.Network, data *dataset) {
	fmt.Println("Testing...")
	startTime := time.Now()
	_, _, err := net.Test(data, bs.SquaredError(true), bs.CorrectHighest)
	if err != nil {
		panic(err.Error())
	}
	fmt.Println("Done testing! It took", time.Since(startTime).Seconds(), "seconds")
}

func save(net *bs.Network) {
	fmt.Println("Saving...")
	if err := net.Save(path, true); err != nil {
		panic(err.Error())
	}
	fmt.Println("Done!")
}

func load() (net *bs.Network) {
	fmt.Println("Loading...")
	types := map[string]bs.Operator{
		"conv-1":          operators.Convolution(nil, optimizers.GradientDescent()),
		"pool-1":          operators.AvgPool(nil),
		"pool-1 logistic": operators.Logistic(),
		"output neurons":  operators.Neurons(optimizers.GradientDescent()),
		"output logistic": operators.Logistic(),
	}
	// no auxiliary information necessary
	var err error
	if net, err = bs.Load(path, types, nil); err != nil {
		panic(err.Error())
	}
	fmt.Println("Done!")

	return
}

func initNet() (net *bs.Network) {
	net = new(bs.Network)
	l, err := net.Add("inputs", nil, imgSize)
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

	return net
}

func main() {
	trainData, err := data(trainFile)
	if err != nil {
		panic(err.Error())
	}

	testData, err := data(testFile)
	if err != nil {
		panic(err.Error())
	}

	net := initNet()
	// net := load()
	train(net, trainData, testData)
	save(net)
}
