package badstudent

import (
	"github.com/pkg/errors"
)

// Note: Correct should likely be removed; it is not used.

// Correct adjusts the weights in the network, according to the provided arguments.
// If 'saveChanges' is true, the adjustments will not be implemented immediately,
// and will instead wait until AddWeights is called.
func (net *Network) Correct(inputs, targets []float64, saveChanges bool) (cost float64, outs []float64, err error) {

	if outs, err = net.GetOutputs(inputs); err != nil {
		err = errors.Wrapf(err, "Getting outputs failed\n")
		return
	}

	if err = net.getDeltas(targets); err != nil {
		err = errors.Wrapf(err, "Getting deltas failed\n")
		return
	}

	if err = net.adjust(saveChanges); err != nil {
		err = errors.Wrapf(err, "Adjusting weights failed\n")
		return
	}

	cost = net.cf.Cost(outs, targets)
	net.iter++
	return
}

// Datum is a simple wrapper used to send training samples to the Network
type Datum struct {
	// Inputs is the input of the network. It must have the same size as that of the
	// network's inputs.
	Inputs []float64

	// Outputs is the expected output of the network, given the input.
	//
	// For recurrent models, providing nil (or length 0) can be used to signify that
	// the outputs are not significant.
	Outputs []float64
}

// Fits indicates whether or not a given Datum's dimensions match those of the
// Network, allowing it to be used for training or testing.
func (d Datum) Fits(net *Network) bool {
	return len(d.Inputs) == net.InputSize() && ((len(d.Outputs) == 0 && net.hasDelay) || len(d.Outputs) == net.OutputSize())
}

// DataSupplier is the primary method of providing datasets to the Network, either
// for training or testing.
type DataSupplier interface {
	// Get returns the next piece of data, given the current iteration.
	Get(int) (Datum, error)

	// BatchEnded returns whether or not the most recent batch has ended, given the
	// current iteration. To not use batching, BatchEnded should always return true
	// (effective mini-batch size of 1).
	//
	// BatchEnded will be called after the last Datum in the batch has been
	// retrieved. It will not be called if the DataSupplier is being used for
	// testing.
	BatchEnded(int) bool

	// DoneTesting indicates whether or not the testing process has finished. This
	// will only be called if the DataSupplier is actually used for providing
	// testing data.
	//
	// DoneTesting is called at the same point as BatchEnded would be.
	DoneTesting(int) bool
}

// Sequential builds upon DataSupplier, adding extra functionality for recurrent
// Networks.
type Sequential interface {
	DataSupplier

	// SetEnded returns whether or not the recurrent sequence has finished, given
	// the current iteration. The DataSupplier methods BatchEnded and DoneTesting
	// will only be called when SetEnded returns true.
	//
	// SetEnded is always immediately followed by BatchEnded or DoneTesting.
	// SetEnded can also cause DoneTesting to be ignored or delayed if not the end
	// of a sequence.
	SetEnded(int) bool
}

// A wrapper for sending back the progress of the training or testing
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

type TrainArgs struct {
	TrainData DataSupplier

	// TestData is the source of cross-validation data while training. This can be
	// nil if ShouldTest is also nil
	TestData DataSupplier

	// ShouldTest indicates whether or not testing should be done before the current
	// iteration. For recurrent Networks, this will only be called in between
	// sequences (modulus might not always work as intended).
	ShouldTest func(int) bool

	// SendStatus indicates whether or not to send back general information about
	// the status of the training since the last time 'true' was returned.
	// SendStatus can be left nil to represent an unconditional false.
	//
	// 'true' will be ignored on iteration 0.
	SendStatus func(int) bool

	// improvement note: A method for Network could be added that gives the previous
	// cost

	// RunCondition will be called at each successive iteration to determine if
	// training should continue. Training will stop if 'false' is returned.
	RunCondition func(int) bool

	// IsCorrect returns whether or not the network outputs are correct, given the
	// target outputs. In order, it is given: outputs; targets.
	//
	// The length of both provided slices is guaranteed to be equal.
	IsCorrect func([]float64, []float64) bool

	// Update is how testing and status updates are returned. If both ShouldTest
	// and SendData are nil, then Update can also be left nil.
	Update func(Result)
}

func (net *Network) Train(args TrainArgs) error {
	// handle error cases and set defaults
	var trainSeq Sequential
	{
		if args.Update == nil {
			args.Update = func(r Result) {}
		}

		if args.TrainData == nil {
			return errors.Errorf("TrainData is nil")
		}

		var ok bool
		if trainSeq, ok = args.TrainData.(Sequential); net.hasDelay && !ok {
			return errors.Errorf("Net has delay; TrainData is not sequential")
		}

		if args.TestData == nil {
			if args.ShouldTest != nil {
				return errors.Errorf("TestData is nil but ShouldTest is not")
			} else {
				args.ShouldTest = func(i int) bool { return false }
			}
		} else if _, ok = args.TestData.(Sequential); net.hasDelay && !ok {
			return errors.Errorf("Net has delay; TestData is not sequential")
		}

		if args.SendStatus == nil {
			args.SendStatus = func(i int) bool { return false }
		}

		if args.RunCondition == nil {
			return errors.Errorf("RunCondition is nil")
		}

		if args.IsCorrect == nil {
			args.IsCorrect = func(a, b []float64) bool { return false }
		}
	}

	// This does overwrite any correction done independently by Correct, but that
	// should (hopefully) not matter
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
					return errors.Wrapf(err, "Testing on iteration %d failed\n", net.iter)
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
			return errors.Wrapf(err, "Failed to get training data on iteration %d\n", net.iter)
		} else if !d.Fits(net) {
			return errors.Errorf("Training data for iteration %d does not fit Network", net.iter)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return errors.Wrapf(err, "Failed to get Network outputs on iteration %d\n", net.iter)
		}

		var cost float64
		var correct bool
		if len(d.Outputs) != 0 { // will always be true for non-recurrent
			cost = net.cf.Cost(outs, d.Outputs)
			correct = args.IsCorrect(outs, d.Outputs)
		}

		endBatch := args.TrainData.BatchEnded(net.iter)

		if !net.hasDelay {
			if err := net.getDeltas(d.Outputs); err != nil {
				return errors.Errorf("Failed to get network deltas on iteration %d\n", net.iter)
			}

			saveChanges := net.hasSavedChanges || !endBatch
			if err = net.adjust(saveChanges); err != nil {
				return errors.Wrapf(err, "Failed to adjust network on iteration %d\n", net.iter)
			}

			if endBatch && net.hasSavedChanges {
				net.AddWeights()
			}
		} else {
			targets = append(targets, d.Outputs)

			if trainSeq.SetEnded(net.iter) {
				if err := net.adjustRecurrent(targets, !(endBatch || batchNext)); err != nil {
					return errors.Wrapf(err, "Failed to adjust recurrent after end of sequence (iteration %d)\n", net.iter)
				}

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

func (net *Network) Test(data DataSupplier, isCorrect func([]float64, []float64) bool) (float64, float64, error) {
	var ok bool
	var dataSeq Sequential
	if dataSeq, ok = data.(Sequential); net.hasDelay && !ok {
		return 0, 0, errors.Errorf("Network has delay but data is not sequential")
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
			return 0, 0, errors.Wrapf(err, "Failed to get test sample %d\n", testSize)
		} else if !d.Fits(net) {
			return 0, 0, errors.Errorf("Test sample %d does not fit Network dimensions\n", testSize)
		}

		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			return 0, 0, errors.Wrapf(err, "Failed to get Network outputs with test sample %d\n", testSize)
		}

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

// Data converts a 3D dataset of float64 to a DataSupplier, which can be used for
// training or testing. dataset indexing is: [data index][inputs, outputs][values]
//
// N.B.: Data does not check if the data fit a certain network; that will be done
// during training/testing
func Data(dataset [][][]float64, batchSize int) (DataSupplier, error) {
	d := dataset
	if len(d) == 0 {
		return nil, errors.Errorf("dataset has no data (len == 0)")
	} else if batchSize < 1 {
		return nil, errors.Errorf("batch size must be >= 1 (%d)", batchSize)
	}

	// check we won't get indexes out of bounds
	for i := range d {
		if len(d[i]) < 2 {
			return nil, errors.Errorf("dataset lacks required data at index %d (len([%d]) < 2)", i, i)
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

// SeqData converts a 3D dataset of float64 to a DataSupplier that is also
// Sequential. SeqData runs Data before attaching an additional method to satisfy
// Sequential, so arguments must follow the same format.
func SeqData(dataset [][][]float64, batchSize, setSize int) (DataSupplier, error) {
	if setSize < 1 {
		return nil, errors.Errorf("setSize must be >= 1 (%d)", setSize)
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
