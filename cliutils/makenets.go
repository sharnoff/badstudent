package cliutils

import (
	"github.com/pkg/errors"

	"github.com/sharnoff/badstudent"
	"github.com/sharnoff/badstudent/operators"
	"github.com/sharnoff/badstudent/operators/optimizers"

	"fmt"
	"bufio"
	"io"
	"strconv"
	"strings"
)

// given format:
// name
// operator as string
// optimizer for operator, if it needs it. There is never a blank line
// name
// operator
// etc...
func GetLoadTypes(r io.Reader) (map[string]badstudent.Operator, error) {
	sc := bufio.NewScanner(r)

	getOpt := func() (operators.Optimizer, error) {
		if !sc.Scan() {
			return nil, errors.Errorf("Couldn't get types, operator given without required optimizer")
		}

		switch sc.Text() {
		case "gradient descent":
			return optimizers.GradientDescent(), nil
		default:
			return nil, errors.Errorf("Couldn't get types, unknown optimizer")
		}
	}

	ops := make(map[string]badstudent.Operator)

	for sc.Scan() {
		name := sc.Text()
		if ops[name] != nil {
			return nil, errors.Errorf("Couldn't get types, duplicate names given")
		}

		if !sc.Scan() {
			return nil, errors.Errorf("Couldn't get types, name given without operator")
		}

		switch sc.Text() {
		case "convolution":
			opt, err := getOpt()
			if err != nil {
				return nil, err
			}
			ops[name] = operators.Convolution(nil, opt)
		case "neurons":
			opt, err := getOpt()
			if err != nil {
				return nil, err
			}
			ops[name] = operators.Neurons(opt)
		case "logistic":
			ops[name] = operators.Logistic()
		case "avg pool":
			ops[name] = operators.AvgPool(nil)
		case "max pool":
			ops[name] = operators.MaxPool(nil)
		case "leaky max pool":
			ops[name] = operators.LeakyMaxPool(nil, 0)
		default:
			return nil, errors.Errorf("Couldn't get types, unknown operator string (%d)", sc.Text())
		}
	}

	return ops, nil
}

// Queries the given input (usually a command line) in order to create a badstudent.Network
//
// given: an input method, output method
// if 'w' is nil, MakeNet() will not output instructions
//
// returns the newly-constructed Network,
// a string that satisifies 'GetLoadTypes'
func MakeNet(r io.Reader, w io.Writer) (*badstudent.Network, string, error) {
	interacting := (w != nil)

	net := new(badstudent.Network)

	sc := bufio.NewScanner(r)

	printf := func(format string, args ...interface{}) {
		if interacting {
			fmt.Fprintf(w, format, args...)
		}
	}

	println := func(args ...interface{}) {
		if interacting {
			fmt.Fprintln(w, args...)
		}
	}

	types := ""
	layers := make(map[string]*badstudent.Layer)

	help := func() {
		println("Commands:")
		println("\t'add' - start the constructor to add a layer")
		println("\t'finish' - set the outputs to the Network and be done")
		println("\t'quit' | 'q' - quit the constructor without saving any progress")
		println("\t'help' - bring up this menu")
	}

	listOfLayers := func(str string) ([]*badstudent.Layer, error) {
		bySpaces := strings.Split(strings.Trim(str, " "), " ")

		outStrs := make([]string, 0, len(bySpaces))
		quote := ""
		inQuote := false
		for _, str := range bySpaces {
			if str == "" {
				return nil, errors.Errorf("Can't have layer with no name (is there a double space?)")
			} else if str == `"` {
				if inQuote {
					outStrs = append(outStrs, quote + " ")
				}
				inQuote = !inQuote
				continue
			}

			starts := strings.HasPrefix(str, `"`)
			ends := strings.HasSuffix(str, `"`)

			if inQuote {
				if starts {
					return nil, errors.Errorf("Can't get layers, close quote present without space")
				} else if ends {
					outStrs = append(outStrs, quote + " " + str[ : len(str) - 1])
					inQuote = false
					quote = "" // not actually necessary
				} else { // neither starts nor ends
					quote += " " + str
				}
			} else { // inQuote == false
				if starts {
					if !ends { // just starts
						quote = str[1 :]
						inQuote = true
					} else { // starts and ends
						outStrs = append(outStrs, str[1 : len(str) - 1])
					}
				} else if ends { // just ends
					return nil, errors.Errorf("Can' get layers, close quote present without start quote")
				} else {
					outStrs = append(outStrs, str)
				}
			}
		}

		ls := make([]*badstudent.Layer, len(outStrs))

		for i, str := range outStrs {
			str = strings.Replace(str, "\\\"", "\"", -1) // -1 indicates all substrings
			if layers[str] == nil {
				return nil, errors.Errorf("Can't get layers, \"%s\" is not a known layer", str)
			}
			ls[i] = layers[str]
		}

		return ls, nil
	}

	// map of type to description
	knownOps := map[string]string{
		"neurons":        "fully connected layer. Linear activation function",
		"convolution":    "standard convolutional layer. Linear activation function",
		"logistic":       "hyperbolic tangent in range (0, 1)",
		"avg pool":       "average pooling",
		"max pool":       "maximum pooling",
		"leaky max pool": "custom max pool that lets a small amount of each input 'leak'",
	}

	// map of type to description
	knownOpts := map[string]string{
		"gradient descent": "standard, run of the mill, gradient descent. No momentum or anything fancy.",
	}

	// containted here because there is no use in other functions and there
	// are many local variables
	add := func() error {
		var l *badstudent.Layer
		var err error
		var name string
		var size int

		var opStr string
		var op badstudent.Operator

		var optStr string
		var opt operators.Optimizer

		var inputs []*badstudent.Layer

		printf("Welcome to the layer constructor. ")
	Restart:
		// get operator type
		println("What operator would you like to use? (type 'help' to see options)")
		for {
			if !sc.Scan() {
				return errors.Errorf("Scanner.Scan() failed while waiting for a command")
			}

			if sc.Text() == "q" || sc.Text() == "quit" {
				println("Leaving layer constructor.")
				return nil
			} else if sc.Text() == "help" {

				println("Layer types:")
				for name, desc := range knownOps {
					printf("\t%s - %s\n", name, desc)
				}

				continue
			}

			if knownOps[sc.Text()] == "" {
				println("Unknown operator. Please pick another, or type 'help'")
				continue
			}

			opStr = sc.Text()
			break
		}

		// get optimizer, if it needs one
		if opStr == "convolution" || opStr == "neurons" {
			println("What type of optimizer would you like to use? (type 'help' to see options)")
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed while waiting for a command")
				}

				if sc.Text() == "q" || sc.Text() == "quit" {
					println("Leaving layer constructor.")
					return nil
				} else if sc.Text() == "help" {

					println("Optimizer types:")
					for name, desc := range knownOpts {
						printf("\t%s - %s\n", name, desc)
					}

					continue
				}

				if knownOpts[sc.Text()] == "" {
					println("Unknown optimizer. Please pick another, or type 'help'")
					continue
				}

				optStr = sc.Text()
				break
			}

			switch optStr {
			case "gradient descent":
				opt = optimizers.GradientDescent()
			default:
				printf("The optimizer type '%s' slipped through the cracks. Exiting.\n", optStr)
				return errors.Errorf("Previously known optimizer became unknown.")
			}
		}

		// get inputs
		println("Please list all layers to use as inputs. Use identical formatting to shell arguments.")
		println("You may type 'list' to see a list of all current layers. To make an input layer, enter nothing.")
		for {
			if !sc.Scan() {
				return errors.Errorf("Scanner.Scan() failed while waiting for input layers")
			}

			if sc.Text() == "quit" || sc.Text() == "q" {
				println("Leaving layer constructor")
				return nil
			} else if sc.Text() == "list" {
				for name := range layers {
					printf(" - %s")
				}
			}

			// if no inputs given
			if sc.Text() == "" {
				printf("Would you like to add an input layer? (y/n): ")
				addInput, quit, err := QueryTF(sc)
				if err != nil {
					return errors.Wrapf(err, "")
				} else if quit {
					println("Exiting layer constructor.")
					return nil
				}

				if addInput {
					break
				} else {
					printf("Try again for inputs: ")
					continue
				}
			}

			inputs, err = listOfLayers(sc.Text())
			if err != nil {
				println(err.Error())
				println("Try again (or type 'quit')")
			} else {
				break
			}
		}

		// given: len(inputDims), &dimension_in_struct,
		// "formatted %d %d string to provide if the dimension sizes don't match (struct, dimSize)",
		// "string to provide at the start of attempting to get the dimensions"
		// 
		// if 'dimSize' == -1, getDims will not check dimension sizes
		//
		// returns 'true', 'false' if the user requested to restart convolution construction
		// returns 'false', 'true' if the user quit
		// does not return 'true', 'true'
		// returns 'false', 'false' if everything ran as expected
		getDims := func(dimSize int, dims *[]int, notMatch, getStr string) (bool, bool, error) {
			for {
				print(getStr)

				if !sc.Scan() {
					return false, false, errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "quit" || sc.Text() == "q" {
					println("Quitting.")
					return false, true, nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				*dims = make([]int, len(dimStrs))
				for i := range dimStrs {
					if (*dims)[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue
					} else if (*dims)[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue
					}
				}

				if dimSize != -1 && len(*dims) != dimSize {
					printf(notMatch, len(*dims), dimSize)

					printf("Restart construction from input dimensions? (y/n): ")
					restart, _, err := QueryTF(sc)
					if err != nil {
						return true, false, errors.Wrapf(err, "")
					}

					if restart { // && !quit
						return true, false, nil
					} else {
						continue
					}
				}

				break
			}

			return false, false, nil
		}

		// construct the operators
		numInputs := 0
		for i := range inputs {
			numInputs += inputs[i].Size()
		}

		var leakFactor float64 // just for leaky max pool

	GetOperator:
		switch opStr {
		case "logistic":
			size = numInputs
			op = operators.Logistic()
		case "leaky max pool": // falls through into the next case
			printf("Please enter a leak factor (between 0 and 1): ")
			isValid := func(v float64) string {
				if v < 0 || v > 1 {
					return "Please enter a leak factor between 0.0 and 1.0:"
				}

				return ""
			}

			var quit bool
			leakFactor, quit, err = QueryFloat(sc, isValid)
			if err != nil {
				return errors.Wrapf(err, "")
			} else if quit {
				println("Exiting the layer constructor.")
				return nil
			}

			fallthrough
		case "avg pool", "max pool":
			args := new(operators.PoolArgs)

			for {
				// InputDims
				restart, quit, err := getDims(-1, &args.InputDims, "", "Please enter preferred input dimensions: ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Dims
				restart, quit, err = getDims(len(args.InputDims), &args.Dims, "Number of output and input dimensions should match (%d != %d)\n", "Please enter preferred output dimensions: ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Filter
				restart, quit, err = getDims(len(args.InputDims), &args.Filter, "Number of filter and input dimensions should match (%d != %d)\n", "Please enter preferred filter size (for each dimension): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Stride
				restart, quit, err = getDims(len(args.InputDims), &args.Stride, "Number of stride and input dimensions should match (%d != %d)\n", "Please enter preferred stride (include '1' for dimensions with a default stride): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				break
			}

			// get size for later
			size = 1
			for i := range args.Dims {
				size *= args.Dims[i]
			}

			if opStr == "avg pool" {
				op = operators.AvgPool(args)
			} else if opStr == "max pool" {
				op = operators.MaxPool(args)
			} else if opStr == "leaky max pool" {
				op = operators.LeakyMaxPool(args, leakFactor)
			} else {
				printf("The operator type '%s' slipped through the cracks. Exiting.\n", opStr)
				return errors.Errorf("Previously known operator became unknown.")
			}
		case "convolution":
			args := new(operators.ConvArgs)

			for {
				// InputDims
				restart, quit, err := getDims(-1, &args.InputDims, "", "Please enter preferred input dimensions: ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Dims
				restart, quit, err = getDims(len(args.InputDims), &args.Dims, "Number of output and input dimensions should match (%d != %d)\n", "Please enter preferred output dimensions (not inlcuding depth): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Filter
				restart, quit, err = getDims(len(args.InputDims), &args.Filter, "Number of filter and input dimensions should match (%d != %d)\n", "Please enter preferred filter size (for each dimension): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Stride
				restart, quit, err = getDims(len(args.InputDims), &args.Stride, "Number of stride and input dimensions should match (%d != %d)\n", "Please enter preferred stride (include '1' for dimensions with a default stride): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				// Zero padding
				restart, quit, err = getDims(len(args.InputDims), &args.ZeroPadding, "Number of zero-padding and input dimensions should match (%d != %d)\n", "Please enter preferred zero padding (include '0' for dimensions with none): ")
				if err != nil {
					return errors.Wrapf(err, "")
				} else if restart {
					continue
				} else if quit {
					return nil
				}

				printf("Please enter preferred depth: ")
				validDepth := func(d int) string {
					if d < 1 {
						return "Depth should be ≥ 1. Try again: "
					}
					return ""
				}
				args.Depth, quit, err = QueryInt(sc, validDepth)
				if err != nil {
					return errors.Wrapf(err, "")
				} else if quit {
					println("Exiting the layer constructor.")
					return nil
				}

				printf("Please enter whether or not to have biases (y/n): ")
				args.Biases, quit, err = QueryTF(sc)
				if err != nil {
					return errors.Wrapf(err, "")
				} else if quit {
					println("Exiting the layer constructor.")
					return nil
				}

				break
			}

			// get size for later
			size = 1
			for i := range args.Dims {
				size *= args.Dims[i]
			}
			size *= args.Depth

			op = operators.Convolution(args, opt)
		case "neurons":
			printf("Please enter preferred number of outputs (given %d inputs): ", numInputs)
			isValid := func(n int) string {
				if n < 1 {
					return "Size should be ≥ 1. Try again: "
				}

				return ""
			}

			var quit bool
			if size, quit, err = QueryInt(sc, isValid); err != nil {
				return errors.Wrapf(err, "")
			} else if quit {
				println("Exiting the layer constructor.")
				return nil
			}

			op = operators.Neurons(opt)
		default:
			printf("The operator type '%s' slipped through the cracks. Exiting.\n", opStr)
			return errors.Errorf("Previously known operator became unknown.")
		}

		// get name of layer
		printf("Enter a name for this layer: ")
		for {
			if !sc.Scan() {
				return errors.Errorf("Scanner.Scan() failed while waiting for a command")
			}

			if sc.Text() == "" {
				printf("Can't make a layer with no name. Enter a name for this layer: ")
			} else if strings.Contains(sc.Text(), `"`) {
				printf("Can't make layer that contains a double-quote. Enter a differnet name for this layer: ")
			} else if sc.Text() == "list" {
				printf("Can't make a layer with a name of 'list'. Enter a different name for this layer: ")
			} else {
				name = sc.Text()
				break
			}
		}

		// confirm that the layer should be added to the network
		printf("Add layer '%s' to network? (y/n): ", name)
		for {
			if !sc.Scan() {
				return errors.Errorf("Scanner.Scan() failed while waiting for confirmation")
			}

			if sc.Text() == "y" {
				break
			} else if sc.Text() == "n" {
				printf("Try again? (y/n): ")
				for {
					if !sc.Scan() {
						return errors.Errorf("Scanner.Scan() failed")
					}

					if sc.Text() == "y" {
						goto Restart
					} else if sc.Text() == "n" {
						return nil
					} else {
						printf("Please enter 'y' or 'n': ")
						continue
					}
				}
				goto GetOperator
			} else {
				printf("Please enter 'y' or 'n': ")
				continue
			}
		}

		// add layer to network
		if l, err = net.Add(name, op, size, inputs...); err != nil {
			println(err.Error())
			printf("Try again? (y/n): ")
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "y" {
					goto Restart
				} else if sc.Text() == "n" {
					return nil
				} else {
					printf("Please enter 'y' or 'n': ")
					continue
				}
			}
		}

		layers[name] = l
		types += name + "\n" + opStr + "\n"
		if opt != nil {
			types += optStr + "\n"
		}

		return nil
	}

	// returns true if it was not quit
	finish := func() (bool, error) {
		println("Please list which layers should be outputs:")
		for {

			if !sc.Scan() {
				return false, errors.Errorf("Scanner.Scan() failed while trying to finish")
			}

			if sc.Text() == "quit" || sc.Text() == "q" {
				return true, nil
			}

			outs, err := listOfLayers(sc.Text())
			if err != nil {
				println(err.Error())
				println("Please try again.")
				continue
			}

			if err := net.SetOutputs(outs...); err != nil {
				println(err.Error())
				println("Please try again.")
				continue
			}

			return false, nil
		}
	}

	println("Welcome to the Network constructor. Here's what you can do:")
	help()
	for {
		printf("Pick a command: ")

		if !sc.Scan() {
			return nil, "", errors.Errorf("Scanner.Scan() failed while waiting for a command")
		}

		switch sc.Text() {
		case "add":
			add()
		case "finish":
			if quit, err := finish(); !quit {
				if err == nil {
					println("Finished making the network!")
					// subtract 1 from the end of 'types' because we don't want the final new line
					return net, types[ : len(types) - 1], nil
				}

				return nil, "", errors.Wrapf(err, "")
			}
		case "quit", "q":
			println("Quitting.")
			return nil, "", errors.Errorf("Quit")
		case "help":
			help()
		default:
			printf("Unknonwn command. ")
		}
	}
}
