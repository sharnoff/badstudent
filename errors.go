package badstudent

// Error is a wrapper for specific types of errors for which there is no additional information
// necessary. These errors are defined as global variables, and can all follow the form: 
type Error struct{ string }

func (err Error) Error() string {
	return err.string
}

// These are the global errors that may be returned or panicked.
var (
	ErrRegisterWrongType = Error{"Type is not recognized"}
	ErrRegisterNilReturn = Error{"Function return is nil"}
)

// NilArgError documents errors resulting from certain arguments provided to a function being nil.
type NilArgError struct{ string }

func (err NilArgError) Error() string {
	return err.string + " is nil"
}
