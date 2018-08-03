package badstudent

import (
	"github.com/pkg/errors"
	"math"
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

	rangeCostDeriv := func(start, end int, add func(int, float64)) error {
		return cf.Deriv(net.outputs.getValues(false), targets, start, end, add)
	}

	for i, in := range net.inputs.nodes {
		if err = in.getDeltas(rangeCostDeriv, false); err != nil { // deltasMatter = false
			err = errors.Wrapf(err, "Couldn't correct network, getting deltas of network input %v (#%d) failed\n", in, i)
		}
	}

	for i, out := range net.outputs.nodes {
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

// The simple package used to send training samples to the Network
type Datum struct {
	Inputs  []float64
	Outputs []float64
}

func (d Datum) fits(net *Network) bool {
	return len(d.Inputs) == net.inputs.size() && len(d.Outputs) == net.outputs.size()
}

// A wrapper for sending back the progress of the training or testing
type Result struct {
	// The iteration the result is being sent before
	Iteration int

	// Average cost, from CostFunc
	Avg float64

	// The percent correct, as per IsCorrect
	// Out of 100
	Percent float64

	// The result is either from a test or a status update
	IsTest bool
}

// Simply a way to condense the large list of arguments to *Network.Train()
type TrainArgs struct {
	// The source for all of the training data for the Network
	//
	// sends a piece of data for each 'true' it is sent.
	// both channels will be pre-made with no buffers
	//
	// the sending of a 'false' indicates that that
	// there will be no more requests for data
	//
	// if there is an error, the channel of Data should be closed
	// else, it will not be registered
	//
	// should set the provided *bool - 'moreData' - to false if there is no more data
	// otherwise, it may deadlock
	Data func(chan bool, chan Datum, *bool, *error)

	// The source for all of the testing data for the Network
	//
	// effectively identical to 'Data'
	// will only send 'false' if there has been an error
	//
	// the channel of data should be closed once there
	// are no more data to test
	TestData func(chan bool, chan Datum, *bool, *error)

	// The number of testing data the network should test on before that iteration
	// Will not test if 0
	//
	// if equal to nil (or not given), will never test (allowing TestData to be nil)
	//
	// the results of testing are sent back through Results,
	// with IsTest = true
	ShouldTest func(int) int

	// Whether or not to send back training status
	// if equal to nil, will never send back data
	//
	// sends before training on that iteration,
	// will not send at iteration 0 (because there has been no training)
	SendStatus func(int) bool

	// The number of training samples to be run before applying the changes to weights
	// also determines the number of training samples requested from
	//
	// defaults to a batch size of 1 (no changes saved)
	//
	// returns: whether or not the next iteration is the start of a batch; whether or not the network should delay changes
	Batch func(int) (bool, bool)

	// Whether or not the network should keep training (will keep training if true)
	// arguments: number of iterations over individual data, previous error (cost)*
	//
	// *will be given NaN if no recorded previous error (iteration 0)
	RunCondition func(int, float64) bool

	// Returns what the learning rate to train the network should be at that iteration
	// arguments: number of iterations over individual data, previous error (cost)*
	//
	// *will be given NaN on iteration 0 because of no previously recorded error
	LearningRate func(int, float64) float64

	// given outputs, targets; should return whether or not the outputs are correct
	// is not required. Defaults to CorrectRound()
	IsCorrect func([]float64, []float64) bool

	// calculates the cost / error for the network on that datum
	// used for both training and testing
	// is not required. Defaults to SquaredError()
	CostFunc CostFunction

	// Where the results of testing and status checking are sent
	// can be provided as nil
	//
	// Train() will return error if Results is nil and
	// SendStatus or ShouldTest return true
	Results chan Result

	// Err should not be nil, and *Err should be nil
	// this is where (how) any error encoutered while training
	// will be returned
	Err *error
}

var default_CostFunc = SquaredError(false)
var default_IsCorrect = CorrectRound
var default_Batch = func(iteration int) (bool, bool) { return true, false }

// Trains the network, given the provided arguments.
// For information on how changes in arguments affect how the Network trains, see TrainArgs
func (net *Network) Train(args TrainArgs) {

	canSend := true
	if args.Results != nil {
		defer close(args.Results)
	}

	// error checking
	{
		if args.Err == nil {
			return
		} else if *args.Err != nil {
			return
		} else if args.Data == nil {
			*args.Err = errors.Errorf("Can't *Network.Train(), no provided data (args.Train == nil)")
			return
		} else if args.RunCondition == nil {
			*args.Err = errors.Errorf("Can't *Network.Train(), no provided run condition (args.RunCondition == nil)")
			return
		} else if args.LearningRate == nil {
			*args.Err = errors.Errorf("Can't *Network.Train(), no provided learning rate (args.LearningRate == nil)")
		}
	}

	// setting defaults
	{
		if args.CostFunc == nil {
			args.CostFunc = default_CostFunc
		}

		if args.IsCorrect == nil {
			args.IsCorrect = default_IsCorrect
		}

		if args.Batch == nil {
			args.Batch = default_Batch
		}

		if args.ShouldTest == nil {
			args.ShouldTest = func(iteration int) int {
				return 0
			}
		}
	}

	var iteration int
	var runningAvgCost, runningPercent float64
	var statusSize int // the amount of data that the status update reports on
	var lastCost float64 = math.NaN()

	var fetchData chan bool = make(chan bool)
	var dataSrc chan Datum = make(chan Datum)
	var moreData bool = true
	var dataErr error

	go args.Data(fetchData, dataSrc, &moreData, &dataErr)

	// whether or not Data() has returned - dataSrc is closed or
	// fetchData has already sent 'false'
	var finishedSafely bool
	defer func() {
		if !finishedSafely {
			fetchData <- false
			close(fetchData)
		}
	}()

	// declared here for the sake of efficiency.
	// does not persist from one iteration to the next
	var saveChanges bool
	var newBatch bool

	var changesDelayed bool

	for args.RunCondition(iteration, lastCost) {

		if args.SendStatus(iteration) && iteration != 0 {
			if !canSend {
				*args.Err = errors.Errorf("Can't *Network.Train(), tried to send status when can't send results (args.Results == nil) (iteration = %d)", iteration)
				return
			}

			runningPercent /= float64(statusSize)
			runningAvgCost /= float64(statusSize)

			args.Results <- Result{iteration, runningAvgCost, runningPercent, false}

			runningPercent, runningAvgCost = 0.0, 0.0
			statusSize = 0
		}

		if amount := args.ShouldTest(iteration); amount != 0 {
			if !canSend {
				*args.Err = errors.Errorf("Can't *Network.Train(), tried to test when can't send results (args.Results == nil) (iteration = %d)", iteration)
				return
			}

			avg, percent, err := net.Test(args.TestData, args.CostFunc, args.IsCorrect, amount)
			if err != nil {
				*args.Err = errors.Wrapf(err, "Couldn't *Network.Train(), testing before iteration %d failed\n", iteration)
				return
			}

			args.Results <- Result{iteration, avg, percent, true}
		}

		if newBatch, saveChanges = args.Batch(iteration); newBatch && iteration != 0 {
			if changesDelayed {
				if err := net.AddWeights(); err != nil {
					*args.Err = errors.Wrapf(err, "Can't *Network.Train(), adding weights from start of new batch on iteration %d failed\n", iteration)
					return
				}

				changesDelayed = false
			}
		}

		if saveChanges {
			changesDelayed = true
		}

		// get data and correct network weights
		var cost float64
		var outs []float64
		var correct bool
		{
			if !moreData {
				finishedSafely = true
				break
			}

			fetchData <- true

			d, ok := <-dataSrc
			if dataErr != nil {
				*args.Err = errors.Wrapf(dataErr, "Can't *Network.Train(), fetching data on iteration %d failed\n", iteration)
				finishedSafely = true
				return
			} else if !ok {
				*args.Err = errors.Errorf("Can't *Network.Train(), data source channel closed without error on iteration %d\n", iteration)
				finishedSafely = true
				return
			}

			if !d.fits(net) {
				*args.Err = errors.Errorf("Can't *Network.Train(), provided Datum on iteration %d does not fit network", iteration)
			}

			var err error
			cost, outs, err = net.Correct(d.Inputs, d.Outputs, args.LearningRate(iteration, lastCost), args.CostFunc, saveChanges)
			if err != nil {
				*args.Err = errors.Wrapf(err, "Couldn't *Network.Train(), correction failed on iteration %d\n", iteration)
				return
			}

			correct = args.IsCorrect(outs, d.Outputs)
		}

		{
			if correct {
				runningPercent += 100
			}
			runningAvgCost += cost
		}

		iteration++
		statusSize++
		lastCost = cost
	}

	// finish up before returning
	{
		if changesDelayed {
			if err := net.AddWeights(); err != nil {
				*args.Err = errors.Wrapf(err, "Couldn't *Network.Train(), adding weights after finishing training (iteration %d) failed\n", iteration)
				return
			}
		}

		if args.SendStatus(iteration) && iteration != 0 {
			if !canSend {
				*args.Err = errors.Errorf("Can't *Network.Train(), tried to send status after training when can't send results (args.Results == nil) (iteration = %d)", iteration)
				return
			}

			runningPercent /= float64(statusSize)
			runningAvgCost /= float64(statusSize)

			args.Results <- Result{iteration, runningAvgCost, runningPercent, false}
		}

		if amount := args.ShouldTest(iteration); amount != 0 {
			if !canSend {
				*args.Err = errors.Errorf("Can't *Network.Train(), tried to test after training when can't send results (args.Results == nil) (iteration = %d)", iteration)
				return
			}

			avg, percent, err := net.Test(args.TestData, args.CostFunc, args.IsCorrect, amount)
			if err != nil {
				*args.Err = errors.Wrapf(err, "Couldn't *Network.Train(), testing after training (iteration %d) failed\n", iteration)
				return
			}

			args.Results <- Result{iteration, avg, percent, true}
		}
	}

	return
}

// Tests the network on the provided data, using 'cf' and 'isCorrect' to return average cost, percent correct
//
// 'amount' is the quantity of test data that should be requested
//
// the responsibility is on Test() to send 'false' to data, not data to close its channel
// for more information on 'data', see TrainArgs.TestData
//
// returns average cost and percent correct (out of 100)
func (net *Network) Test(data func(chan bool, chan Datum, *bool, *error), cf CostFunction, isCorrect func([]float64, []float64) bool, amount int) (float64, float64, error) {

	var fetchData chan bool = make(chan bool)
	var dataSrc chan Datum = make(chan Datum)
	var dataErr error
	var moreData bool = true

	go data(fetchData, dataSrc, &moreData, &dataErr)

	var finishedSafely bool
	defer func() {
		if !finishedSafely {
			fetchData <- false
			close(fetchData)
		}
	}()

	var avgCost, percent float64

	i := 0 // used for reporting errors
	for {
		if !moreData {
			finishedSafely = true
			break
		}

		fetchData <- true

		d, ok := <-dataSrc
		if dataErr != nil {
			finishedSafely = true
			return 0, 0, errors.Wrapf(dataErr, "Can't *Network.Test(), fetching data for test iteration %d failed\n", i)
		} else if !ok {
			finishedSafely = true
			return 0, 0, errors.Errorf("Can't *Network.Test(), data source channel closed without error on test iteration %d\n", i)
		}

		if !d.fits(net) {
			return 0, 0, errors.Errorf("Can't *Network.Test(), provided Datum on test iteration %d does not fit network", i)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't *Network.Test(), could not get outputs of network on test iteraiton %d\n", i)
		}

		c, err := cf.Cost(outs, d.Outputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Couldn't *Network.Test(), could not calculate cost of outputs on test iteration %d\n", i)
		}
		avgCost += c

		if isCorrect(outs, d.Outputs) {
			percent += 100.0
		}

		i++
	}

	avgCost /= float64(i)
	percent /= float64(i)

	return avgCost, percent, nil
}

// Returns a function that satisfies TrainArgs.Data or TrainArgs.TestData, given a list of data
//
// data[n] should have length 2
// data[n][0] is inputs, data[n][1] is outputs
//
// 'loop' should be true for training samples, false for testing
func DataCh(data [][][]float64, loop bool) (func(chan bool, chan Datum, *bool, *error), error) {

	// check that all of the data are okay (but does not check if they fit the network)
	for i := range data {
		if len(data[i]) != 2 {
			return nil, errors.Errorf("Can't get DataCh, len(data[%d]) != 2 (%d)", i, len(data[i]))
		}
	}

	return func(request chan bool, dataSrc chan Datum, moreData *bool, err *error) {
		defer close(dataSrc)

		for {
			for i := range data {
				req := <-request
				if !req {
					return
				}

				dataSrc <- Datum{data[i][0], data[i][1]}
			}

			if !loop {
				*moreData = false
				break
			}
		}

		return
	}, nil
}
