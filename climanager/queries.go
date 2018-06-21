package climanager

import (
	"github.com/pkg/errors"

	"bufio"
	"strconv"
	"fmt"
)

// returns user input of true/false, whether or not user quits
// returns error if scanner runs out
//
// in case of other return states (error or quit), booleans default to 'false'
func QueryTF(sc *bufio.Scanner) (bool, bool, error) {
	for {
		if !sc.Scan() {
			return false, false, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return false, true, nil
		}

		if sc.Text() == "y" || sc.Text() == "yes" {
			return true, false, nil
		} else if sc.Text() == "n"  || sc.Text() == "no" {
			return false, false, nil
		} else {
			fmt.Print("Please enter 'y' or 'n': ")
		}
	}
}

// gets an integer from user input
// input is provided by 'sc' and the integer returned will be in the range specified by 'isValid'
//
// 'isValid' returns an error string if the given integer is out of bounds
// if 'isValid' returns an empty string, that integer will be returned by QueryInt
// the error message from 'isValid' will be printed as: `fmt.Print(error-message)`
//
// returns 'true' only if the user quits (enters 'quit' or 'q')
// returns an error only if input runs out -- if sc.Scan() returns 'false'
func QueryInt(sc *bufio.Scanner, isValid func(int) string) (int, bool, error) {
	for {
		if !sc.Scan() {
			return 0, false, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return 0, true, nil
		}
		
		if v, err := strconv.Atoi(sc.Text()); err != nil {
			fmt.Print("Please enter an integer: ")
		} else if errMsg := isValid(v); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return v, false, nil
		}
	}
}

// gets a float from user input
// functionally identical to QueryInt, but for float64
func QueryFloat(sc *bufio.Scanner, isValid func(float64) string) (float64, bool, error) {
	for {
		if !sc.Scan() {
			return 0, false, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return 0, true, nil
		}
		
		if v, err := strconv.ParseFloat(sc.Text(), 64); err != nil {
			fmt.Print("Please enter an floating point number: ")
		} else if errMsg := isValid(v); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return v, false, nil
		}
	}
}

// gets a string from user input
// functionally identical to QueryInt, but for strings.
func QueryString(sc *bufio.Scanner, isValid func(string) string) (string, bool, error) {
	for {
		if !sc.Scan() {
			return "", false, errors.Errorf("Scanner.Scan() failed.")
		}

		if sc.Text() == "quit" || sc.Text() == "q" {
			return "", true, nil
		}
		
		if errMsg := isValid(sc.Text()); errMsg != "" {
			fmt.Print(errMsg)
		} else {
			return sc.Text(), false, nil
		}
	}
}
