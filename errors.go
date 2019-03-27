package badstudent

import (
	"fmt"
)

// errors.go contains some of the errors found in badstudent. Other specific error types can be
// found near the functions that use them.

// Error is a wrapper for specific types of errors for which there is no additional information
// necessary. These errors are defined as global variables, and can all follow the form:
type Error struct{ string }

func (err Error) Error() string {
	return err.string
}

// These are the global errors that may be returned or panicked.
var (
	ErrRegisterWrongType = Error{"Type is not recognized"}
	ErrRegisterNilReturn = Error{"Registered function return is nil"}

	ErrDifferentNetworkInput  = Error{"Output Node belongs to a different network"}
	ErrDifferentNetworkOutput = Error{"One or more input Node(s) belongs to different network"}

	ErrNetFinalized       = Error{"Network has already been finalized"}
	ErrNetNotFinalized    = Error{"Network has not been finalized"}
	ErrNilNet             = Error{"Method called on nil Network"}
	ErrNilInputNode       = Error{"One or more input Node(s) is nil"}
	ErrInvalidOperator    = Error{"Operator is invalid (Does not implement Layer or Elementwise)"}
	ErrNoInputs           = Error{"Network has no inputs"}
	ErrNoOutputs          = Error{"No outputs have been given"}
	ErrIsInput            = Error{"Output Node is an input"}
	ErrOutputHasDelay     = Error{"Output Node has non-zero delay"}
	ErrDuplicateOutput    = Error{"Output Node is provided twice (or more)"}
	ErrNoDefaultOptimizer = Error{"No default optimizer has been set"}
	ErrHPNameTaken        = Error{"HyperParameter name has already been registered with Node/Network"}
	ErrNoHPToReplace      = Error{"HyperParameter name has not already been registered with Node/Network"}

	ErrInvalidDelay     = Error{"Delay must be ≥ 0"}
	ErrDelayPlaceholder = Error{"Cannot set delay of a placeholder Node"}
	ErrNotPlaceholder   = Error{"Cannot replace non-placeholder Node"}
	ErrDelayInput       = Error{"Cannot set delay of an input Node"}

	ErrNoHP          = Error{"No HyperParameter by given name"}
	ErrNoInputValues = Error{"Node is an input; does not have input values."}

	ErrNegativeIter = Error{"Given iteration is less than zero."}

	ErrFailedCommand = Error{"Graphviz dot command failed."}

	ErrTrainNotSequential = Error{"Network has delay but training data is not sequential"}
	ErrTestNotSequential  = Error{"Network has delay but testing data is not sequential"}
	ErrShouldTestButNil   = Error{"TestData is nil but ShouldTest is not"}
	ErrNoData             = Error{"Given dataset has no data (len=0)"}
	ErrSmallBatchSize     = Error{"Given batch size is less than 1"}
	ErrSmallSetSize       = Error{"Given set size is less than 1"}
)

// NilArgError documents errors resulting from certain arguments provided to a function being nil.
type NilArgError struct{ string }

func (err NilArgError) Error() string {
	return err.string + " is nil"
}

// SizeMismatchError records errors where the given size of a piece of a slice or similar object is
// not the same as what was expected.
type SizeMismatchError struct {
	Expected, Given int

	Description string
}

func (err SizeMismatchError) Error() string {
	return fmt.Sprintf("Size of %s differs from expected. Expected size: %d, given: %d", err.Description, err.Expected, err.Given)
}
