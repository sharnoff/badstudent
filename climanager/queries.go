package climanager

import (
	"github.com/pkg/errors"

	"bufio"
	"strconv"
	"fmt"
)

// returns true if 'true', false if 'false'
// returns error if recieves 'quit' or scanner runs out
func QueryTF(sc *bufio.Scanner) (bool, error) {
	for {
		if !sc.Scan() {
			return false, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return false, errors.Errorf("Recieved quit signal.")
		}

		if sc.Text() == "y" || sc.Text() == "yes" {
			return true, nil
		} else if sc.Text() == "n"  || sc.Text() == "no" {
			return false, nil
		} else {
			fmt.Print("Please enter 'y' or 'n': ")
		}
	}
}

// 'isValid' should return an error message if the given integer is out of bounds
// if 'isValid' returns an empty string, that integer will be returned by QueryInt
// the error message from 'isValid' will used as: `fmt.Print("error-message")`
//
// returns error if it recieves 'quit' or scanner runs out
func QueryInt(sc *bufio.Scanner, isValid func(int) string) (int, error) {
	for {
		if !sc.Scan() {
			return 0, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return 0, errors.Errorf("Recieved quit signal")
		}
		
		if v, err := strconv.Atoi(sc.Text()); err != nil {
			fmt.Print("Please enter an integer: ")
		} else if errMsg := isValid(v); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return v, nil
		}
	}
}

// 'isValid' should return an error message if the given string is not an option
// if 'isValid' returns an empty string, the input string will be returned by QueryString
// the error message from 'isValid' will be used as: `fmt.Print("error-message")`
//
// returns error if it recieves 'quit' or scanner runs out
func QueryString(sc *bufio.Scanner, isValid func(string) string) (string, error) {
	for {
		if !sc.Scan() {
			return "", errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return "", errors.Errorf("Recieved quit signal")
		}
		
		if errMsg := isValid(sc.Text()); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return sc.Text(), nil
		}
	}
}

// 'isValid' should return an error message if the given float is out of bounds
// if 'isValid' returns an empty string, that float will be returned by QueryInt
// the error message from 'isValid' will used as: `fmt.Print("error-message")`
//
// returns error if it recieves 'quit' or scanner runs out
func QueryFloat(sc *bufio.Scanner, isValid func(float64) string) (float64, error) {
	for {
		if !sc.Scan() {
			return 0, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return 0, errors.Errorf("Recieved quit signal")
		}
		
		if v, err := strconv.ParseFloat(sc.Text(), 64); err != nil {
			fmt.Print("Please enter an floating point number: ")
		} else if errMsg := isValid(v); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return v, nil
		}
	}
}
