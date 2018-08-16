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

// The primary method of supplying data to the network -- for training OR testing
type DataSupplier interface {
	// Whether or not the data being supplied is for a recurrent-network
	//
	// Will only be called once, before any data is requested
	IsSequential() bool

	// Returns the next bit of data
	Get() (Datum, error)

	// Only for recurrent networks: true iff there is no more data in the most recent set
	SetEnded() bool

	// Whether or not the most recent batch has ended
	// For an effective batch size of 1, this should always return true.
	BatchEnded() bool

	// only called if this is supplying test data
	// Whether or not the necessary amount of testing has been done
	DoneTesting() bool
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
	// The source of data used for training the network
	TrainData DataSupplier

	// The source of data used for ongoing dev/validation of the network while training
	// Can be nil if and only if ShouldTest is nil
	TestData DataSupplier

	// If omitted, TestData will recieve no use; TestData can be nil if this is nil
	// Will be given the current iteration
	//
	// Testing is performed before training. Ex: if on iteration 'n', a batch is completed,
	// then the next opportunity to test is on iteration 'n + 1', before the start of the
	// next batch
	ShouldTest func(int) bool

	// Can be omitted to represent an unconditional false
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

		if args.TrainData == nil {
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

	var betweenSets bool = true

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
			if net.hasDelay && !betweenSets {
				*args.Err = errors.Errorf("Testing for recurrent networks must be done between sets")
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

		betweenSets = false

		d, err := args.TrainData.Get()
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
		batch := args.TrainData.BatchEnded()

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
			}
		} else { // if the network has delay
			targets = append(targets, d.Outputs)
			learningRates = append(learningRates, α)

			if args.TrainData.SetEnded() {
				if err := net.adjustRecurrent(targets, args.CostFunc, learningRates, !batch); err != nil {
					*args.Err = errors.Wrapf(err, "Failed to adjust recurrent model after end of set\n")
					return
				}

				targets = nil
				learningRates = nil
				betweenSets = true
			} else if batch { // but not set ended
				*args.Err = errors.Errorf("Batching for recurrent models must align with sets")
				return
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
			net.ClearDelays()
		}
	}

	return
}

// args.TestData, args.CostFunc, args.IsCorrect
//
// Information about arguments to Test can be found in TrainArgs
// Will return: average cost; average fraction correct; any errors
func (net *Network) Test(data DataSupplier, costFunc CostFunction, isCorrect func([]float64, []float64) bool) (float64, float64, error) {
	if net.hasDelay && !data.IsSequential() {
		return 0, 0, errors.Errorf("Network has delay but data is not sequential")
	}

	var avgCost, avgCorrect float64
	var testSize int

	defer net.ClearDelays()

	for {
		d, err := data.Get()
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

		if data.SetEnded() {
			net.ClearDelays()
		}

		testSize++

		if data.DoneTesting() {
			break
		}
	}

	if testSize != 0 {
		avgCost /= float64(testSize)
		avgCorrect /= float64(testSize)
	}

	return avgCost, avgCorrect, nil
}

type internalSupplier struct {
	get         func() (Datum, error)
	setEnded    func() bool
	batchEnded  func() bool
	doneTesting func() bool
}

func (s internalSupplier) IsSequential() bool {
	return s.setEnded != nil
}

func (s internalSupplier) Get() (Datum, error) {
	return s.get()
}

func (s internalSupplier) SetEnded() bool {
	return s.setEnded()
}

func (s internalSupplier) BatchEnded() bool {
	return s.batchEnded()
}

func (s internalSupplier) DoneTesting() bool {
	return s.doneTesting()
}

// dataset should either be a slice of [][]float64 (where [0] is inputs,
// [1] is outputs), or a slice of Datum
//
// for setEnded and batchEnded:
// both are given an index in the provided data
// both are called after the data at the specified index has been used
//
// setEnded can be set to nil if the data is not sequential
func Data(dataset interface{}, setEnded, batchEnded func(int) bool) (DataSupplier, error) {
	var l int
	if reflect.ValueOf(dataset).Kind() != reflect.Slice {
		return nil, errors.Errorf("dataset must be a compatible type (%T is not)", dataset)
	} else if l = reflect.ValueOf(dataset).Len(); l == 0 {
		return nil, errors.Errorf("dataset must have len > 0")
	}

	var get func(int) Datum

	if d, ok := dataset.([]Datum); ok {
		get = func(index int) Datum {
			return d[index]
		}
	} else if d, ok := dataset.([][][]float64); ok {
		get = func(index int) Datum {
			return Datum{d[index][0], d[index][1]}
		}
	} else {
		return nil, errors.Errorf("dataset is not of a compatible type (%T)", dataset)
	}

	var is internalSupplier

	var index int
	is.get = func() (Datum, error) {
		d := get(index)
		index++
		if index >= l {
			index = 0
		}
		return d, nil
	}

	if setEnded != nil {
		is.setEnded = func() bool {
			return setEnded(index)
		}
	}

	is.batchEnded = func() bool {
		return batchEnded(index)
	}

	is.doneTesting = func() bool {
		return index == 0
	}

	return is, nil
}
