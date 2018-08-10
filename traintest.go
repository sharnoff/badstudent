package badstudent

import (
	"github.com/pkg/errors"
	"math"
	"reflect"
)

// Adjusts the weights in the network, according to the provided arguments
// if 'saveChanges' is true, the adjustments will not be implemented immediately
//
// 'learningRate' determines the strength of the changes
// 'cf' is the CostFunction that will be optimized for
func (net *Network) Correct(inputs, targets []float64, learningRate float64, cf CostFunction, saveChanges bool) (cost float64, outs []float64, err error) {
	outs, err = net.GetOutputs(inputs)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't correct network, getting outputs failed\n")
		return
	}

	if err = net.getDeltas(targets, cf); err != nil {
		err = errors.Wrapf(err, "Failed to get deltas of Network\n")
		return
	}

	if err = net.adjust(learningRate, saveChanges); err != nil {
		err = errors.Wrapf(err, "Failed to adjust weights of Network\n")
		return
	}

	cost, err = cf.Cost(outs, targets)
	if err != nil {
		err = errors.Wrapf(err, "Couldn't calculate cost after correcting network, SquaredError() failed\n")
		return
	}

	return
}

// The simple package used to send training samples to the Network
type Datum struct {
	Inputs  []float64
	Outputs []float64
}

func (d Datum) fits(net *Network) bool {
	return len(d.Inputs) == net.InputSize() && len(d.Outputs) == net.OutputSize()
}

// A wrapper for sending back the progress of the training or testing
type Result struct {
	// The iteration the result is being sent before
	Iteration int

	// Average cost, from CostFunc
	Cost float64

	// The fraction correct, as per IsCorrect() from TrainArgs
	// 0 → 1
	Correct float64

	// The result is either from a test or a status update
	IsTest bool
}

type TrainArgs struct {
	// Will be called at each successive iteration to fetch the next sample of
	// training data.
	//
	// Expected returns are: sample of training data; whether or not this sample is
	// the last in a batch; a message with a reason to stop training, if any.
	//
	// To perform stochastic gradient descent, simply return 'true' each time.
	//
	// A note about RNNs: For training recurrent neural networks with the same arguments as
	// feed-forward networks, batch boundaries are used to signify the end of a sequence.
	// Additionally, the
	Data func() (Datum, bool, error)

	// Can be omitted if ShouldTest is omitted.
	// Will run every time ShouldTest() returns true
	//
	// Should return true only while the last sample is being given.
	//
	// The results of the testing will be sent through Results
	TestData func() (Datum, bool, error)

	// If omitted, TestData() will recieve no use
	// Will be given the current iteration
	//
	// Testing is performed before training. Ex: if on iteration 'n', a batch is completed,
	// then the next opportunity to test is on iteration 'n + 1', before the start of the
	// next batch
	ShouldTest func(int) bool

	// Can be omitted
	//
	// If 'true' is returned, general information about the status of the training
	// since the last time 'true' was returned will be sent through Results
	//
	// Will ignore 'true' on iteration 0
	SendStatus func(int) bool

	// Will be called at each sucessive iteration to determine if training should
	// continue. Training will stop if 'false' is returned.
	//
	// Will be given the iteration (starting at zero) and the error (cost) of the
	// previous iteration, as determined by CostFunc.
	// Iteration values start at 0, and the previous error will be NaN for that 0th
	// iteration.
	RunCondition func(int, float64) bool

	// Will be called at each successive iteration to determine the learning rate to
	// be used in adjusting the values of the weights in the Network.
	//
	// Will be given the iteration (starting at zero) and the error (cost) of the
	// previous iteration, as determined by CostFunc.
	// Iteration values start at 0, and the previous error will be NaN for that 0th
	// iteration.
	LearningRate func(int, float64) float64

	// If omitted, Results will always give 'Correct' of 0
	//
	// Will be given: network outputs; target outputs. The length of both will be the
	// same.
	IsCorrect func([]float64, []float64) bool

	// If omitted, will default to SquaredError(false)
	CostFunc CostFunction

	// Can be omitted.
	// Information about the training -- testing results and status updates -- will be sent
	// through this channel
	Results chan Result

	// If there is an error, this is where it will be returned.
	// Train will not run if this is nil.
	Err *error
}

// Trains the network
func (net *Network) Train(args TrainArgs) {
	// handle error cases and set defaults
	{
		if args.Results == nil {
			args.Results = make(chan Result)

			go func() {
				for _ = range args.Results {
				}
			}()
		}

		defer close(args.Results)

		if args.Err == nil {
			return
		} else if *args.Err != nil {
			return
		}

		if args.Data == nil {
			*args.Err = errors.Errorf("args.Data cannot be nil; there must be training data supplied to the network")
			return
		}

		if args.TestData == nil {
			if args.ShouldTest != nil {
				*args.Err = errors.Errorf("If args.TestData is nil, so must args.ShouldTest")
				return
			} else {
				args.ShouldTest = func(iteration int) bool {
					return false
				}
			}
		}

		if args.SendStatus == nil {
			args.SendStatus = func(iteration int) bool {
				return false
			}
		}

		if args.RunCondition == nil {
			*args.Err = errors.Errorf("args.RunCondition cannot be nil")
			return
		}

		if args.LearningRate == nil {
			*args.Err = errors.Errorf("args.LearningRate cannot be nil; there must be a learning rate in order to train")
			return
		}

		if args.IsCorrect == nil {
			args.IsCorrect = func(a, b []float64) bool {
				return false
			}
		}

		if args.CostFunc == nil {
			args.CostFunc = SquaredError(false)
		}
	}

	var iteration int
	var lastCost float64 = math.NaN()

	var statusCost, statusCorrect float64
	var statusSize int

	// used only for training recurrent neural networks
	var targets [][]float64
	var learningRates []float64

	var betweenBatches bool = true

	// for run condition (actually slightly farther down)
	for {
		if args.SendStatus(iteration) && iteration != 0 {
			statusCost /= float64(statusSize)
			statusCorrect /= float64(statusSize)

			args.Results <- Result{
				Iteration: iteration,
				Cost:      statusCost,
				Correct:   statusCorrect,
				IsTest:    false,
			}

			statusCost, statusCorrect = 0, 0
			statusSize = 0
		}

		if args.ShouldTest(iteration) {
			if net.hasDelay && !betweenBatches {
				*args.Err = errors.Errorf("Testing for recurrent networks must be done between batches")
				return
			}

			cost, correct, err := net.Test(args.TestData, args.CostFunc, args.IsCorrect)
			if err != nil {
				*args.Err = errors.Wrapf(err, "Testing failed on iteration %d\n", iteration)
				return
			}

			args.Results <- Result{
				Iteration: iteration,
				Cost:      cost,
				Correct:   correct,
				IsTest:    true,
			}
		}

		if !args.RunCondition(iteration, lastCost) {
			break
		}

		betweenBatches = false

		d, batch, err := args.Data()
		if err != nil {
			*args.Err = errors.Wrapf(err, "Failed to get training data on iteration %d\n", d)
			return
		} else if !d.fits(net) {
			*args.Err = errors.Errorf("Training data recieved for iteration %d does not fit Network", iteration)
			return
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			*args.Err = errors.Wrapf(err, "Failed to get network outputs on iteration %d\n", iteration)
			return
		}

		cost, err := args.CostFunc.Cost(outs, d.Outputs)
		if err != nil {
			*args.Err = errors.Wrapf(err, "Failed to get cost of outputs on iteration %d\n", iteration)
		}

		correct := args.IsCorrect(outs, d.Outputs)

		α := args.LearningRate(iteration, lastCost)
		if !net.hasDelay { // if the network is 'normal'
			if err := net.getDeltas(d.Outputs, args.CostFunc); err != nil {
				*args.Err = errors.Wrapf(err, "Failed to get network deltas on iteration %d\n", iteration)
				return
			}

			saveChanges := net.hasSavedChanges || !batch
			if err := net.adjust(α, saveChanges); err != nil {
				*args.Err = errors.Wrapf(err, "Failed to adjust network on iteration %d\n", iteration)
			}

			if batch {
				if err := net.AddWeights(); err != nil {
					*args.Err = errors.Wrapf(err, "Failed to add weights after iteration %d\n", iteration)
					return
				}

				betweenBatches = true
			}
		} else {
			targets = append(targets, d.Outputs)
			learningRates = append(learningRates, α)

			if batch {
				if err := net.adjustRecurrent(targets, args.CostFunc, learningRates); err != nil {
					*args.Err = errors.Wrapf(err, "Failed to apply recurrent adjustment methods to network on iteration %d\n", iteration)
					return
				}

				targets = nil
				learningRates = nil

				betweenBatches = true
			}
		}

		statusCost += cost
		if correct {
			statusCorrect += 1.0
		}
		statusSize++

		lastCost = cost
		iteration++
	}

	// finish up before returning
	{
		if net.hasSavedChanges {
			if err := net.AddWeights(); err != nil {
				*args.Err = errors.Wrapf(err, "Failed to add weights to network post-training\n")
				return
			}
		}

		if net.hasDelay {
			net.clearDelays()
		}
	}

	return
}

// args.TestData, args.CostFunc, args.IsCorrect
//
// Information about arguments to Test can be found in TrainArgs
// Will return: average cost; average fraction correct; any errors
func (net *Network) Test(data func() (Datum, bool, error), costFunc CostFunction, isCorrect func([]float64, []float64) bool) (float64, float64, error) {
	var avgCost, avgCorrect float64
	var testSize int

	for {
		d, last, err := data()
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Failed to get sample #%d\n", testSize)
		}

		if !d.fits(net) {
			return 0, 0, errors.Errorf("Test sample #%d does not fit network dimensions\n", testSize)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Failed to get outputs of network with sample #%d\n", testSize)
		}

		cost, err := costFunc.Cost(outs, d.Outputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Failed to get cost of outputs with sample #%d\n", testSize)
		}
		avgCost += cost

		if isCorrect(outs, d.Outputs) {
			avgCorrect += 1
		}

		testSize++
		if last {
			avgCost /= float64(testSize)
			avgCorrect /= float64(testSize)

			return avgCost, avgCorrect, nil
		}
	}
}

// possible types for dataset are: [][][]float64; []Datum; chan [][]float64; chan Datum
//
// if isTest is true, then endBatch will be ignored -- can be nil
func Data(dataset interface{}, endBatch func(int) bool) (func() (Datum, bool, error), error) {
	v := reflect.ValueOf(dataset)
	if v.Kind() != reflect.Chan && v.Kind() != reflect.Slice {
		return nil, errors.Errorf("%T is not a compatible type", dataset)
	}

	toDatum := func(d [][]float64) Datum {
		return Datum{d[0], d[1]}
	}

	var get func() Datum

	if v.Kind() == reflect.Chan {
		if d, ok := dataset.(chan [][]float64); ok {
			get = func() Datum {
				return toDatum(<-d)
			}
		} else if d, ok := dataset.(chan Datum); ok {
			get = func() Datum {
				return <-d
			}
		} else {
			return nil, errors.Errorf("%T is not a compatible type", dataset)
		}
	} else { // must be a Slice
		if v.Len() == 0 {
			return nil, errors.Errorf("Must be supplied data; len(dataset) == 0")
		}

		var index = 0
		if d, ok := dataset.([][][]float64); ok {
			get = func() Datum {
				i := index
				index++
				if index >= len(d) {
					index = 0
				}
				return toDatum(d[i])
			}
		} else if d, ok := dataset.([]Datum); ok {
			get = func() Datum {
				i := index
				index++
				if index >= len(d) {
					index = 0
				}
				return d[i]
			}
		} else {
			return nil, errors.Errorf("%T is not a compatible type", dataset)
		}
	}

	var iteration int

	return func() (Datum, bool, error) {
		iteration++
		return get(), endBatch(iteration - 1), nil
	}, nil
}
