package badstudent

// setError sets the Network's stored error to the error provided. If net.panicErrors is true,
// setError will additionally panic the error it is given.
func (net *Network) setError(e error) {
	net.err = e
	if net.panicErrors {
		panic(e)
	}
}

// Nodes returns the list of all Nodes in the Network, sorted by ID such that Nodes()[n] has id=n.
// The slice that Nodes returns is a copy; it can be modified freely but will not update if more
// Nodes are added to the Network.
func (net *Network) Nodes() []*Node {
	ns := make([]*Node, len(net.nodesByID))
	copy(ns, net.nodesByID)
	return ns
}

// ResetIter resets the Network's tracked number of iterations to the provided value. This could be
// done to bring HyperParameters that are dependent upon iterations back to an earlier state. The
// given value will usually be zero. ResetIter will return ErrNegativeIter if the iteration given
// is less than zero.
func (net *Network) ResetIter(iter int) error {
	if iter < 0 {
		return ErrNegativeIter
	}

	net.longIter = iter
	return nil
}

// InputSize returns the total number of expected input values to the Network. If the Network has
// not been finalized yet, InputSize will return -1.
func (net *Network) InputSize() int {
	if net.stat < finalized {
		return -1
	}

	return net.inputs.size()
}

// OutputSize returns the total number of expected output values to the Network. If the Network has
// not been finalized yet, OutputSize will return -1.
func (net *Network) OutputSize() int {
	if net.stat < finalized {
		return -1
	}

	return net.outputs.size()
}

// CurrentInputs returns a copy of the current input values to the Network. CurrentInputs returns
// nil if the Network has not been finalized yet.
func (net *Network) CurrentInputs() []float64 {
	if net.stat < finalized {
		return nil
	}

	return net.inputs.getValues(true)
}

// SetInputs sets the inputs of the Network to the provided values. If the Network has not been
// finalized, ErrNetNotFinalized will be returned (or panicked if PanicErrors() has been called).
// Else, if the number of inputs does not equal the total size of the inputs (given by
// InputSize()), type SizeMismatchError will be returned.
func (net *Network) SetInputs(inputs []float64) error {
	if net.stat < finalized {
		if net.panicErrors {
			panic(ErrNetNotFinalized)
		}

		return ErrNetNotFinalized
	}

	err := net.inputs.setValues(inputs)
	if err != nil {
		err = SizeMismatchError{net.inputs.size(), len(inputs), "inputs"}

		if net.panicErrors {
			panic(err)
		}

		return err
	}

	net.stat = finalized
	return err
}

// GetOutputs returns a copy of the Network's output values for the given inputs. SetInputs() will
// be called regardless of whether or not the given inputs are actually the current inputs. There
// are several error conditions:
//	(0) If the Network has not been finalized: ErrNetNotFinalized,
//	(1) If the number of inputs doesn't match the total size: type SizeMismatchError,
// If PanicErrors() has been called, error conditions will be panicked, not returned.
func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if err := net.SetInputs(inputs); err != nil {
		return nil, err
	}

	// evaluate should only return ErrNetNotFinalized, but we check anyways for future-proofing.
	if err := net.evaluate(); err != nil {
		return nil, err
	}

	return net.outputs.getValues(true), nil
}

// ChangeCost changes the CostFunction of the Network, after it has been finalized. This allows
// different CostFunctions for training and final model evaluation. If cf is nil, ChangeCost will
// panic with type NilArgError.
func (net *Network) ChangeCost(cf CostFunction) *Network {
	if cf == nil {
		panic(NilArgError{"CostFunction"})
	}

	net.cf = cf
	return net
}

// Error returns any errors encountered while constructing the Network, particularly while creating
// the architecture. This method will always return nil after the Network has been SUCESSFULLY
// finalized.
func (net *Network) Error() error {
	return net.err
}
