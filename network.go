package badstudent

import (
	"github.com/pkg/errors"
)

// Returns the number of expected input values to the Network
// Returns -1 if the network has not been completed yet
func (net *Network) InputSize() int {
	if net.stat < finalized {
		return -1
	}

	return net.inputs.size()
}

// Returns the number of values that the network outputs
// Returns -1 if the network has not been completed yet
func (net *Network) OutputSize() int {
	if net.stat < finalized {
		return -1
	}

	return net.outputs.size()
}

// Returns the current values of the network inputs.
// Returns nil if the network has not been completed yet
func (net *Network) CurrentInputs() []float64 {
	if net.stat < finalized {
		return nil
	}

	return net.inputs.getValues(true)
}

// sets the inputs of the network to the provided values
// returns an error if the length of the provided values doesn't
// match the size of the network inputs
func (net *Network) SetInputs(inputs []float64) error {
	if net.stat < finalized {
		return errors.Errorf("Network is not complete")
	}

	err := net.inputs.setValues(inputs)
	if err == nil {
		net.stat = finalized
	}
	return err
}

// Returns a copy of the Network's output values for the given inputs
// Returns an error if given the wrong number of inputs
func (net *Network) GetOutputs(inputs []float64) ([]float64, error) {
	if err := net.SetInputs(inputs); err != nil {
		return nil, errors.Wrapf(err, "Setting inputs failed\n")
	}

	if err := net.evaluate(false); err != nil {
		return nil, errors.Wrapf(err, "Failed to evaluate all Nodes in Network\n")
	}

	return net.outputs.getValues(true), nil
}

// SetCost changes the CostFunction of the Network, post-Finalization. This allows
// different CostFunctions for training and final model evaluation.
func (net *Network) SetCost(cf CostFunction) *Network {
	if net.err != nil {
		return net
	}

	net.cf = cf
	return net
}

// Error returns any errors encountered while constructing the network, particularly
// while shaping the architecture. This method will always return nil after the
// Network has been Finalized.
func (net *Network) Error() error {
	return net.err
}
