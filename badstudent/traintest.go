package badstudent

import (
	"github.com/pkg/errors"
	"math"
)

// returns a copy of the output values of the network, given the inputs
// returns error if the number of given inputs doesn't match the number of inputs to the network
func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if len(inputs) != len(net.inputs) {
		return nil, errors.Errorf("Can't get outputs, len(inputs) != len(net.inputs) (%d != %d)", len(inputs), len(net.inputs))
	}

	copy(net.inputs, inputs)
	for _, in := range net.inLayers {
		in.inputsChanged()
	}

	for i, out := range net.outLayers {
		if err := out.evaluate(); err != nil {
			return nil, errors.Wrapf(err, "Can't get outputs, network output layer %v (#%d) failed to evaluate\n", out, i)
		}
	}

	dupe := make([]float64, len(net.outputs))
	copy(dupe, net.outputs)
	return dupe, nil
}

// expects that the given CostFunc will not be nil
func (net *Network) Correct(inputs, targets []float64, learningRate float64, cf CostFunction, saveChanges bool) (cost float64, outs []float64, err error) {
	outs, err = net.GetOutputs(inputs)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't correct network, getting outputs failed\n")
		return
	}

	rangeCostDeriv := func(start, end int, add func(int, float64)) error {
		return cf.Deriv(net.outputs, targets, start, end, add)
	}

	for i, in := range net.inLayers {
		if err = in.getDeltas(rangeCostDeriv); err != nil {
			err = errors.Wrapf(err, "Couldn't correct network, getting deltas of network input %v (#%d) failed\n", in, i)
		}
	}

	for i, out := range net.outLayers {
		if err = out.adjust(learningRate, saveChanges); err != nil {
			err = errors.Wrapf(err, "Couldn't correct network, adjusting failed for output %v (#%d)\n", out, i)
		}
	}

	cost, err = cf.Cost(outs, targets)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't calculate cost after correcting network, SquaredError() failed\n")
		return
	}

	return
}

func (net *Network) addWeights() error {
	for i, out := range net.outLayers {
		if err := out.addWeights(); err != nil {
			return errors.Wrapf(err, "Couldn't add weights of network, output layer %v (#%d) failed to add weights\n", out, i)
		}
	}

	return nil
}

type Datum struct {
	Inputs  []float64
	Outputs []float64
}

func (d Datum) fits(net *Network) bool {
	return len(d.Inputs) == len(net.inputs) && len(d.Outputs) == len(net.outputs)
}

type TrainArgs struct {
	// sends requested amount of data over the provided channel
	// args:
	//	batchSize int
	//		the amount of data that *Network.Train() will expect to receive
	//	iteration int
	// 		a counter for the number of data that have already been used.
	//		This does not work with variable dataset sizes
	//	dataSrc chan struct{Datum, bool}
	//		a channel to send data on
	//		unbuffered, sends block until recieved by *Network.Train()
	//		Epoch is only true if this is the last Datum in the set
	//	err *error
	//		A place for the function to return error4
	Data func(int, int, chan struct {
		Datum
		Epoch bool
	}, *error)

	// sends the testing data (dev set) over the channel
	// can only be nil if TrainBeforeTest == 0
	// args:
	//	dataSrc chan Datum
	//		a channel to send the testing data on
	//		unbuffered, sends block until recieved by *Network.Test()
	//	err *error
	//		A place for the function to return error
	TestData func(chan Datum, *error)

	// the number of training data that *Network.Train() will optimize for before testing
	// must be >= 0. If equal to 0, it will never test (allowing TestData to be nil)
	TrainBeforeTest int

	// the number of training samples to be run before applying the changes to weights
	// also determines the number of training samples requested from Data()
	// should be >= 1, will perform Stochastic Gradient Descent if equal to 1
	BatchSize int

	// whether or not the network should keep training
	// arguments: number of iterations over individual data, number of epochs, previous error (cost)*
	//
	// *will be given NaN if no recorded previous error
	RunCondition func(int, int, float64) bool

	// given outputs, targets; should return whether or not the outputs are correct
	// is not required. Defaults to CorrectRound()
	IsCorrect func([]float64, []float64) bool

	// calculates the cost / error for the network on that datum
	// used for both training and testing
	// is not required. Defaults to SquaredError()
	CostFunc CostFunction

	Results chan struct {
		// the average error/cost from the past batch/epoch/test
		Avg float64

		// the percent correct from the past batch/epoch/test,
		// as determined by IsCorrect()
		Percent float64

		// Epoch is true if the result is for the entire epoch
		// IsTest is true if the result is from test data
		// If neither is true, the result is from a (mini-)batch
		Epoch, IsTest bool
	}

	Err *error
}

var default_CostFunc CostFunction = SquaredError(false)
var default_IsCorrect func([]float64, []float64) bool = CorrectRound 

func (net *Network) Train(args TrainArgs, learningRate float64) {
	res := args.Results
	errPtr := args.Err

	// error checking
	{
		if errPtr == nil {
			if res != nil {
				close(res)
			}
			return
		} else if *errPtr != nil {
			if res != nil {
				close(res)
			}
			return
		} else if res == nil {
			if errPtr != nil && *errPtr == nil {
				*errPtr = errors.Errorf("Can't *Network.Train(), provided Results channel is nil")
			}
			return
		} else if args.Data == nil {
			*errPtr = errors.Errorf("Can't *Network.Train(), provided Data function is nil")
			close(res)
			return
		} else if args.TestData == nil && args.TrainBeforeTest != 0 {
			*errPtr = errors.Errorf("Can't *Network.Train(), provided TestData function is nil")
			close(res)
			return
		} else if args.TrainBeforeTest < 0 {
			*errPtr = errors.Errorf("Can't *Network.Train(), provided 'TrainBeforeTest' < 0 (%d)", args.TrainBeforeTest)
			close(res)
			return
		} else if args.BatchSize < 1 {
			*errPtr = errors.Errorf("Can't *Network.Train(), provided BatchSize < 1. (%d) SGD can be performed by setting BatchSize to 1", args.BatchSize)
			close(res)
			return
		} else if args.RunCondition == nil {
			*errPtr = errors.Errorf("Can't *Network.Train(), provided 'RunCondition' function is nil")
			close(res)
			return
		}
	}

	defer close(res)

	if args.IsCorrect == nil {
		args.IsCorrect = CorrectRound
	}

	if args.CostFunc == nil {
		args.CostFunc = SquaredError(false)
	} 

	var iteration, epoch, epochSize int
	var epochErr, epochPercent float64
	var batchErr, batchPercent float64
	lastErr := math.NaN()

	var dataSrc chan struct {
		Datum
		Epoch bool
	}
	var dataSrcErr error

	saveChanges := (args.BatchSize != 1)

	for {
		if !args.RunCondition(iteration, epoch, lastErr) {
			break
		}

		if args.TrainBeforeTest > 0 && iteration % args.TrainBeforeTest == 0 {
			avg, percent, err := net.Test(args.TestData, args.CostFunc, args.IsCorrect)
			if err != nil {
				*errPtr = errors.Wrapf(err, "Couldn't *Network.Train(), testing before iteration %d (epoch %d) failed\n", iteration, epoch)
				return
			}

			res <- struct {
				Avg, Percent  float64
				Epoch, IsTest bool
			}{avg, percent, false, true}
		}

		if iteration % args.BatchSize == 0 {
			dataSrc = make(chan struct {
				Datum
				Epoch bool
			})

			go args.Data(args.BatchSize, iteration, dataSrc, &dataSrcErr)
		}

		d := <-dataSrc
		if dataSrcErr != nil {
			*errPtr = errors.Wrapf(dataSrcErr, "Couldn't *Network.Train(), fetching data on iteration %d failed\n", iteration)
			return
		} else if d.Inputs == nil && d.Outputs == nil && d.Epoch == false { // check to see if the struct has values that aren't initialized
			*errPtr = errors.Errorf("Couldn't *Network.Train(), received empty struct from fetching data on iteration %d\n", iteration)
			return
		} else if !d.fits(net) {
			*errPtr = errors.Errorf("Couldn't *Network.Train(), Datum at iteration %d (epoch %d) had improper dimensions for network (lengths: net inputs - %d, d.Inputs - %d, net outputs - %d, d.Outputs - %d\n", iteration, epoch, len(net.inLayers[0].values), len(d.Inputs), len(net.outLayers[0].values), len(d.Outputs))
			return
		}

		cost, outs, err := net.Correct(d.Inputs, d.Outputs, learningRate, args.CostFunc, saveChanges)
		if err != nil {
			*errPtr = errors.Wrapf(err, "Couldn't *Network.Train(), correction failed (on iteration %d, epoch %d)\n", iteration, epoch)
			return
		}

		epochErr += cost // will be divided when we know how big the epoch is
		batchErr += cost / float64(args.BatchSize)

		correct := args.IsCorrect(outs, d.Outputs)
		if correct {
			epochPercent += 100
			batchPercent += 100 / float64(args.BatchSize)
		}

		lastErr = cost

		iteration++
		epochSize++

		// if it's the end of a batch, send the information back
		if (iteration % args.BatchSize == 0) && args.BatchSize != 1 {
			res <- struct {
				Avg, Percent  float64
				Epoch, IsTest bool
			}{batchErr, batchPercent, false, false}

			batchErr, batchPercent = 0, 0

			if err := net.addWeights(); err != nil {
				*errPtr = errors.Wrapf(err, "Couldn't *Network.Train(), adding weights failed on iteration %d, epoch %d\n", iteration, epoch)
				return
			}
		}

		// if it's the end of the epoch, send the information back
		if d.Epoch {
			epochErr /= float64(epochSize)
			epochPercent /= float64(epochSize)
			epochSize = 0
			epoch++

			res <- struct {
				Avg, Percent  float64
				Epoch, IsTest bool
			}{epochErr, epochPercent, true, false}

			epochErr, epochPercent = 0.0, 0.0
		}
	}

	// finish up before returning
	{
		shouldAddWeights := false

		if epochSize > 0 {
			epochErr /= float64(epochSize)
			epochPercent /= float64(epochSize)

			res <- struct {
				Avg, Percent  float64
				Epoch, IsTest bool
			}{epochErr, epochPercent, true, false}

			shouldAddWeights = true
		}

		if finalBatchSize := iteration % args.BatchSize; finalBatchSize > 0 {
			batchErr *= float64(args.BatchSize) / float64(finalBatchSize)
			batchPercent *= float64(args.BatchSize) / float64(finalBatchSize)

			res <- struct {
				Avg, Percent  float64
				Epoch, IsTest bool
			}{batchErr, batchPercent, false, false}

			shouldAddWeights = true
		}

		if shouldAddWeights {
			if err := net.addWeights(); err != nil {
				*errPtr = errors.Wrapf(err, "Failed to add weights after training network\n")
				return
			}
		}
	}
	
	return
}

func (net *Network) Test(data func(chan Datum, *error), costFunc CostFunction, isCorrect func([]float64, []float64) bool) (float64, float64, error) {
	dataSrc := make(chan Datum)
	var dataSrcErr error
	go data(dataSrc, &dataSrcErr)

	i := 0
	var avgErr, percentCorrect float64
	for d := range dataSrc {
		if !d.fits(net) {
			return 0, 0, errors.Errorf("Couldn't *Network.Test(), given Datum had improper dimensions (lenghs: net inputs - %d, d.Inputs - %d, net outputs - %d, d.Outputs - d", len(net.inLayers[0].values), len(d.Inputs), len(net.outLayers[0].values), len(d.Outputs))
		}

		if dataSrcErr != nil {
			return 0, 0, errors.Wrapf(dataSrcErr, "Couldn't *Network.Test(), failed to get data for test iteration %d\n", i)
		} else if d.Inputs == nil && d.Outputs == nil {
			return 0, 0, errors.Errorf("Couldn't *Network.Test(), given datum struct has not been initialized for test iteration %d", i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't *Network.Test(), getting outputs for test iteration %d failed\n", i)
		}

		c, err := costFunc.Cost(outs, d.Outputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't *Network.Test(), cost function failed on test iteration %d\n", i)
		}
		avgErr += c

		if isCorrect(outs, d.Outputs) {
			percentCorrect += 100.0
		}

		i++
	}

	avgErr /= float64(i)
	percentCorrect /= float64(i)

	return avgErr, percentCorrect, nil
}

// returns a function that works to satisfy TrainArgs.Data
// data[n] should have length 2
// data[n][0] should be inputs, data[n][1] should be outputs
func TrainCh(data [][][]float64) (func(int, int, chan struct {
	Datum
	Epoch bool
}, *error), error) {

	// check that all of the data are okay
	for i := range data {
		if len(data[i]) != 2 {
			return nil, errors.Errorf("Can't get function to provide to *Network.Train(), len(data[%d]) != 2 (%d)", i, len(data[i]))
		}
	}

	return func(amount, start int, ch chan struct {
		Datum
		Epoch bool
	}, err *error) {

		for i := 0; i < amount; i++ {
			index := (start + i) % len(data)

			var epoch bool
			if index == len(data)-1 {
				epoch = true
			}

			ch <- struct {
				Datum
				Epoch bool
			}{Datum{data[index][0], data[index][1]}, epoch}
		}

		close(ch)
		return
	}, nil
}

// works to satisfy TrainArgs.TestData | *Network.Test(.data)
// data[n] should have length 2
// data[n][0] should be inputs, data[n][1] should be outputs
func TestCh(data [][][]float64) (func(chan Datum, *error), error) {
	// check that all of the data are okay
	for i := range data {
		if len(data[i]) != 2 {
			return nil, errors.Errorf("Can't get function to provide to *Network.Test(), len(data[%d]) != 2 (%d)", i, len(data[i]))
		}
	}

	return func(ch chan Datum, err *error) {
		for _, d := range data {
			ch <- Datum{d[0], d[1]}
		}

		close(ch)
		return
	}, nil
}
