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
	"github.com/sharnoff/badstudent/costfuncs"
	"github.com/sharnoff/badstudent/hyperparams"
	"github.com/sharnoff/badstudent/initializers"
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

func (d *dataset) Get(index int) (bs.Datum, error) {
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

func (d *dataset) BatchEnded(index int) bool {
	return d.index%batchSize == 0
}

func (d *dataset) DoneTesting(index int) bool {
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

const (
	learningRate    float64 = 0.01
	maxIterations   int     = 10000
	batchSize       int     = 10
	testFrequency   int     = 20000
	statusFrequency int     = 1000
)

func train(net *bs.Network, trainData, testData *dataset) {
	up, final := bs.PrintResult()

	args := bs.TrainArgs{
		TrainData:    trainData,
		TestData:     testData,
		ShouldTest:   bs.EndEvery(testFrequency),
		SendStatus:   bs.Every(statusFrequency),
		RunCondition: bs.TrainUntil(maxIterations),
		IsCorrect:    bs.CorrectHighest,
		Update:       up,
	}

	fmt.Println("Starting training...")
	startTime := time.Now()

	// fmt.Println("Iteration, Status Cost, Status Percent Correct, Test Cost, Test Percent Correct")
	if err := net.Train(args); err != nil {
		panic(err.Error())
	}

	final()
	since := time.Since(startTime)
	fmt.Printf("Done training! It took %v seconds (%v minutes)\n", since.Seconds(), since.Minutes())
}

func test(net *bs.Network, data *dataset) {
	fmt.Println("Testing...")
	startTime := time.Now()
	if _, _, err := net.Test(data, bs.CorrectHighest); err != nil {
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
	var err error
	if net, err = bs.Load(path); err != nil {
		panic(err.Error())
	}
	fmt.Println("Done!")

	return
}

func initNet() (net *bs.Network) {
	fmt.Println("Creating...")
	net = new(bs.Network)
	l := net.Add("inputs", nil, imgSize)

	// /*
	l = net.Add("conv-1", operators.Conv().Dims(28, 28).InputDims(28, 28).Filter(5, 5).Stride(1, 1).Pad(2, 2),
		784, l).Init(initializers.VarianceScaling().In().Factor(float64(784)/(5*5)))
	// must initialize with VarianceScaling because of the filter sizes

	l = net.Add("pool-1", operators.AvgPool().Dims(14, 14).InputDims(28, 28).Filter(2, 2),
		196, l)

	l = net.Add("output neurons", operators.Neurons(), 10, l).Opt(optimizers.SGD())
	l = net.Add("output logistic", operators.Logistic(), 10, l)

	net.AddHP("learning-rate", hyperparams.Constant(learningRate))
	net.DefaultInit(initializers.Xavier())

	if err := net.Finalize(costfuncs.MSE(), l); err != nil {
		panic(err.Error())
	}
	// */

	/*
	l = net.Add("hidden neurons", operators.Neurons(), 500, l).Opt(optimizers.SGD())
	l = net.Add("hidden logistic", operators.Logistic(), 500, l)

	l = net.Add("out neurons", operators.Neurons(), 10, l).Opt(optimizers.SGD())
	l = net.Add("out logistic", operators.Logistic(), 10, l)

	net.AddHP("learning-rate", hyperparams.Constant(learningRate))
	net.DefaultInit(initializers.Xavier())

	if err := net.Finalize(costfuncs.MSE(), l); err != nil {
		panic(err.Error())
	}
	*/

	fmt.Println("Done!")

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
