package badstudent

import (
	"fmt"
)

// Datum is a simple type used to send training samples to the Network
type Datum struct {
	// Inputs is the input of the network. It must have the same size as that of the network's
	// inputs.
	Inputs []float64

	// Outputs is the expected output of the network, given the input.
	//
	// For recurrent networks, providing nil (or length 0) can be used to signify that the outputs
	// are not significant, and that the hidden state will be updated to reflect the inputs
	Outputs []float64
}

// Fits indicates whether or not a given Datum's dimensions match those of the Network, allowing it
// to be used for training or testing.
func (d Datum) Fits(net *Network) bool {
	return len(d.Inputs) == net.InputSize() && ((len(d.Outputs) == 0 && net.hasDelay) || len(d.Outputs) == net.OutputSize())
}

// DataSupplier is the primary method of providing datasets to the Network, either for training or
// testing.
type DataSupplier interface {
	// Get returns the next piece of data, given the current iteration.
	Get(int) (Datum, error)

	// BatchEnded returns whether or not the most recent batch has ended, given the current
	// iteration. To not use batching, BatchEnded should always return true (effective mini-batch
	// size of 1).
	//
	// BatchEnded will be called after the last Datum in the batch has been retrieved. It will not
	// be called if the DataSupplier is being used for testing.
	BatchEnded(int) bool

	// DoneTesting indicates whether or not the testing process has finished. This will only be
	// called if the DataSupplier is actually used for providing testing data.
	//
	// DoneTesting is called at the same point as BatchEnded would be.
	DoneTesting(int) bool
}

// Sequential builds upon DataSupplier, adding extra functionality for recurrent Networks.
type Sequential interface {
	DataSupplier

	// SetEnded returns whether or not the recurrent sequence has finished, given the current
	// iteration. The DataSupplier methods BatchEnded and DoneTesting will only be called when
	// SetEnded returns true.
	//
	// SetEnded is always immediately followed by BatchEnded or DoneTesting. SetEnded can also
	// cause DoneTesting to be ignored or delayed if not the end of a sequence.
	SetEnded(int) bool
}

// Result is a wrapper for sending back the progress of the training or testing
type Result struct {
	// The iteration the result is being sent before
	Iteration int

	// Average cost, from the Network's CostFunction
	Cost float64

	// The fraction correct, as per IsCorrect() from TrainArgs
	// 0 → 1
	Correct float64

	// The result is either from a test or a status update
	IsTest bool
}

// TrainArgs serves to allow optional arguments to (*Network).Train()
type TrainArgs struct {
	TrainData DataSupplier

	// TestData is the source of cross-validation data while training. This can be nil if
	// ShouldTest is also nil
	TestData DataSupplier

	// ShouldTest indicates whether or not testing should be done before the current iteration. For
	// recurrent Networks, this will only be called in between sequences (modulus might not always
	// work as intended).
	ShouldTest func(int) bool

	// SendStatus indicates whether or not to send back general information about the status of the
	// training since the last time 'true' was returned. SendStatus can be left nil to represent an
	// unconditional false.
	//
	// 'true' will be ignored on iteration 0.
	SendStatus func(int) bool

	// improvement note: A method for Network could be added that gives the previous
	// cost

	// RunCondition will be called at each successive iteration to determine if training should
	// continue. Training will stop if 'false' is returned.
	RunCondition func(int) bool

	// IsCorrect returns whether or not the network outputs are correct, given the target outputs.
	// In order, it is given: outputs; targets.
	//
	// The length of both provided slices is guaranteed to be equal.
	IsCorrect func([]float64, []float64) bool

	// Update is how testing and status updates are returned. If both ShouldTest and SendData are
	// nil, then Update can also be left nil.
	Update func(Result)
}

// TrainContext provides additional context to training/testing-based errors. Iterations are stored
// across multiple calls to Train()
type TrainContext struct {
	Iteration int
	FromTest  bool
}

// GetDataError wraps errors from DataSupplier.Get() from (*Network).Train() and (*Network).Test()
type GetDataError struct {
	TrainContext

	Err error
}

func (err GetDataError) Error() string {
	var test string
	if err.FromTest {
		test = "test "
	}

	return fmt.Sprintf("Failed to get"+test+"data. Iteration: %d. Error: %v.", err.Iteration, err.Err.Error())
}

// DoesNotFitError results from provided training/testing samples not fitting the dimensions of the
// Network (i.e. number of inputs/outputs doesn't match) Note: This does not extend to cases where
// no outputs are given to recurrent Networks in order to signify that they are inconsequential.
type DoesNotFitError struct {
	TrainContext

	Net *Network
	D   Datum
}

func (err DoesNotFitError) Error() string {
	testData := "Data "
	if err.FromTest {
		testData = "Test data "
	}

	var erroneous string
	if len(err.D.Inputs) != err.Net.InputSize() {
		erroneous += fmt.Sprintf(" Inputs expected %d, got %d.", err.Net.InputSize(), len(err.D.Inputs))
	}

	if len(err.D.Outputs) != 0 && len(err.D.Outputs) != err.Net.OutputSize() {
		erroneous += fmt.Sprintf(" Outputs expected %d, got %d.", err.Net.OutputSize(), len(err.D.Outputs))
	}

	return fmt.Sprintf(testData+"from Iteration %d didn't match Network dimensions.%s", err.Iteration, erroneous)
}

// Train does what it says. It trains the Network following the conditions laid out in the
// arguments provided.
//
// Train has several error conditions:
//	(0) args.Runcondition == nil;
//	(1) args.TrainData == nil;
//	(2) args.TrainData is not sequential but Network has delay;
//	(3) args.TestData is not sequential but Network has delay;
//	(4) args.ShouldTest != nil but args.TestData == nil;
//	(5) Failures to run TrainData.Get() or TestData.Get();
//	(6) Data provided by Get() doesn't fit Network;
// (0) and (1) return type NilArgError, (2) and (3) return ErrTrainNotSequential and
// ErrTestNotSequential, respectively. (4) returns ErrShouldTestButNil, (5) gives type
// GetdataError, and (6) returns type DoesNotFitError.
func (net *Network) Train(args TrainArgs) error {
	// handle error cases and set defaults
	var trainSeq Sequential
	{
		if args.Update == nil {
			args.Update = func(r Result) {}
		}

		if args.TrainData == nil {
			return NilArgError{"TrainData"}
		}

		var ok bool
		if trainSeq, ok = args.TrainData.(Sequential); net.hasDelay && !ok {
			return ErrTrainNotSequential
		}

		if args.TestData == nil {
			if args.ShouldTest != nil {
				return ErrShouldTestButNil
			} else {
				args.ShouldTest = func(i int) bool { return false }
			}
		} else if _, ok = args.TestData.(Sequential); net.hasDelay && !ok {
			return ErrTestNotSequential
		}

		if args.SendStatus == nil {
			args.SendStatus = func(i int) bool { return false }
		}

		if args.RunCondition == nil {
			return NilArgError{"RunCondition"}
		}

		if args.IsCorrect == nil {
			args.IsCorrect = func(a, b []float64) bool { return false }
		}
	}

	net.longIter += net.iter
	net.iter = 0

	var statusCost, statusCorrect float64
	var statusSize int

	// used only for training RNNs
	var targets [][]float64
	var betweenSequences, testNext, batchNext bool = net.hasDelay, false, false // a (very) slight optimization

	// for args.RunCondition() (conditional is embedded farther down)
	for {
		if args.SendStatus(net.iter) && net.iter != 0 {
			r := Result{
				Iteration: net.iter,
				Cost:      statusCost / float64(statusSize),
				Correct:   statusCorrect / float64(statusSize),
				IsTest:    false,
			}

			args.Update(r)

			statusCost, statusCorrect = 0, 0
			statusSize = 0
		}

		if args.ShouldTest(net.iter) || (!betweenSequences && testNext) {
			if net.hasDelay && !betweenSequences {
				testNext = true
			} else {
				cost, correct, err := net.Test(args.TestData, args.IsCorrect)
				if err != nil {
					return err
				}

				r := Result{
					Iteration: net.iter,
					Cost:      cost,
					Correct:   correct,
					IsTest:    true,
				}

				args.Update(r)
			}
		}

		if !args.RunCondition(net.iter) {
			break
		}

		betweenSequences = false

		d, err := args.TrainData.Get(net.iter)
		if err != nil {
			return GetDataError{TrainContext{net.iter, false}, err}
		} else if !d.Fits(net) {
			return DoesNotFitError{TrainContext{net.iter, false}, net, d}
		}

		// GetOutputs will return an error in one of two conditions:
		// (0) If the Network has not been finalized (which we know is false because we already
		// checked that), and (1) if the number of inputs doesn't match Network inputs. This cannot
		// be the case (even with multithreading) because we just checked that.
		//
		// Therefore, we can discard the (im)possible error
		outs, _ := net.GetOutputs(d.Inputs)

		var cost float64
		var correct bool
		if len(d.Outputs) != 0 { // will always be true for non-recurrent
			cost = net.cf.Cost(outs, d.Outputs)
			correct = args.IsCorrect(outs, d.Outputs)
		}

		endBatch := args.TrainData.BatchEnded(net.iter)

		if !net.hasDelay {
			net.getDeltas(d.Outputs)

			// saveChanges = net.hasSavedChanges || !endBatch
			net.adjust(net.hasSavedChanges || !endBatch)

			if endBatch && net.hasSavedChanges {
				net.AddWeights()
			}
		} else {
			targets = append(targets, d.Outputs)

			if trainSeq.SetEnded(net.iter) {

				// saveChanges = (endBatch || batchNext)
				net.adjustRecurrent(targets, !(endBatch || batchNext))

				targets = nil
				betweenSequences = true
				batchNext = false
			} else if endBatch {
				batchNext = true
			}
		}

		statusCost += cost
		if correct {
			statusCorrect += 1.0
		}
		if len(d.Outputs) != 0 {
			statusSize++
		}

		net.iter++
	}

	// finish up before returning
	{
		if net.hasSavedChanges {
			net.AddWeights()
		}

		if net.hasDelay {
			net.ClearDelays()
		}
	}

	return nil
}

// Test will test the Network on the supplied Data and function for determining whether or not the
// outputs are correct. Test returns (in order) the average cost of the outputs and the percent of
// the outputs that are correct.
//
// Test has several possible error conditions:
//	(0) If 'data' is not Sequential, but the Network has delay: ErrTestNotSequential;
//	(1) Failures in data.Get(): type GetDataError;
//	(2) If !data.Get(i).Fits(net): type DoesNotFitError;
// Test also assumes that 'data' is non-nil, and will panic (without a particular error) if that
// interface is nil.
func (net *Network) Test(data DataSupplier, isCorrect func([]float64, []float64) bool) (float64, float64, error) {
	var ok bool
	var dataSeq Sequential
	if dataSeq, ok = data.(Sequential); net.hasDelay && !ok {
		return 0, 0, ErrTestNotSequential
	}

	var avgCost, avgCorrect float64
	var testSize int = 0

	// may result in a superfluous flush
	defer net.ClearDelays()

	// done only refers to RNNs and DoneTesting waiting for SetEnded
	var done bool

	for {
		if data.DoneTesting(testSize) {
			if !net.hasDelay {
				break
			}
			done = true
		}

		if net.hasDelay && dataSeq.SetEnded(testSize) {
			net.ClearDelays()
			if done {
				break
			}
		}

		testSize++

		d, err := data.Get(testSize)
		if err != nil {
			return 0, 0, GetDataError{TrainContext{net.iter, true}, err}
		} else if !d.Fits(net) {
			return 0, 0, DoesNotFitError{TrainContext{net.iter, true}, net, d}
		}

		// for the same reasons as outlined in (*Network).Train(), we can ignore the error output
		// from GetOutputs.
		outs, _ := net.GetOutputs(d.Inputs)

		if len(d.Outputs) == 0 {
			continue
		}

		avgCost += net.cf.Cost(outs, d.Outputs)
		if isCorrect(outs, d.Outputs) {
			avgCorrect += 1
		}
	}

	if testSize != 0 {
		avgCost /= float64(testSize)
		avgCorrect /= float64(testSize)
	}

	return avgCost, avgCorrect, nil
}

type internalSupplier struct {
	get         func(int) (Datum, error)
	batchEnded  func(int) bool
	doneTesting func(int) bool
}

type internalSequential struct {
	internalSupplier
	setEnded func(int) bool
}

func (s internalSupplier) Get(iter int) (Datum, error) {
	return s.get(iter)
}

func (s internalSupplier) BatchEnded(iter int) bool {
	return s.batchEnded(iter)
}

func (s internalSupplier) DoneTesting(iter int) bool {
	return s.doneTesting(iter)
}

func (s internalSequential) SetEnded(iter int) bool {
	return s.setEnded(iter)
}

type DataMissingError struct {
	Index int
	Vs    [][]float64
}

func (err DataMissingError) Error() string {
	return fmt.Sprintf("Dataset missing data at index %d: len(data[%d])=%d, should be 2", err.Index, err.Index, len(err.Vs))
}

// Data converts a 3D dataset of float64 to a DataSupplier, which can be used for training or
// testing. dataset indexing is: [data index][inputs, outputs][values]
//
// N.B.: Data does not check if the data fit a certain network; that will be done during
// training/testing
//
// Data has a few error conditions:
//	(0) If len(dataset) == 0, ErrNoData;
//	(1) If batchSize < 1, ErrSmallBatchSize;
//	(2) If len(dataset[i]) < 2, type DataMissingError;
func Data(dataset [][][]float64, batchSize int) (DataSupplier, error) {
	d := dataset
	if len(d) == 0 {
		return nil, ErrNoData
	} else if batchSize < 1 {
		return nil, ErrSmallBatchSize
	}

	// check we won't get indexes out of bounds
	for i := range d {
		if len(d[i]) < 2 {
			return nil, DataMissingError{i, d[i]}
		}
	}

	is := internalSupplier{
		get: func(iter int) (Datum, error) {
			i := iter % len(dataset)
			return Datum{d[i][0], d[i][1]}, nil
		},
		batchEnded:  EndEvery(batchSize),
		doneTesting: EndEvery(len(dataset)),
	}

	return is, nil
}

// SeqData converts a 3D dataset of float64 to a DataSupplier that is also Sequential. SeqData runs
// Data before attaching an additional method to satisfy Sequential, so arguments must follow the
// same format.
//
// SeqData returns the same errors that Data() returns, and will additionally return
// ErrSmallSetSize if setSize < 1.
func SeqData(dataset [][][]float64, batchSize, setSize int) (DataSupplier, error) {
	if setSize < 1 {
		return nil, ErrSmallSetSize
	}

	ds, err := Data(dataset, batchSize)
	if err != nil {
		return nil, err
	}
	is, _ := ds.(internalSupplier)

	return internalSequential{
		internalSupplier: is,
		setEnded:         EndEvery(setSize),
	}, nil
}
