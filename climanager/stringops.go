package climanager

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
		println("\t'abort' - quit the constructor without saving any progress")
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

	// containted here because there is no use in other functions
	add := func() error {

		help := func() {
			println("Layer types:")
			for name, desc := range knownOps {
				printf("\t%s - %s\n", name, desc)
			}
		}
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

			if sc.Text() == "abort" {
				println("Leaving layer constructor.")
				return nil
			} else if sc.Text() == "help" {
				help()
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

				if sc.Text() == "abort" {
					println("Leaving layer constructor.")
					return nil
				} else if sc.Text() == "help" {
					help()
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
		println("Please list all layers to use as inputs (none if input layer; separate by spaces, refer to by name)")
	GetInputs:
		for {
			if !sc.Scan() {
				return errors.Errorf("Scanner.Scan() failed while waiting for input layers")
			}

			if sc.Text() == "abort" {
				println("Leaving layer constructor")
				return nil
			}

			// if no inputs given
			if sc.Text() == "" {
				printf("Would you like to add an input layer? (y/n): ")
				for {
					if !sc.Scan() {
						return errors.Errorf("Scanner.Scan() failed while waiting for confirmation")
					}

					if sc.Text() == "y" {
						break GetInputs
					} else if sc.Text() == "n" {
						printf("Try again for inputs: ")
						continue GetInputs
					} else {
						printf("Please enter 'y' or 'n': ")
						continue
					}
				}
			}

			inputs, err = listOfLayers(sc.Text())
			if err != nil {
				println(err.Error())
				println("Try again (or type 'abort')")
			} else {
				break
			}
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

			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				leakFactor, err = strconv.ParseFloat(sc.Text(), 64)
				if err != nil {
					println("Please enter a floating-point number:")
					continue
				}

				if leakFactor < 0 || leakFactor > 1 {
					println("Please enter a leak factor between 0.0 and 1.0:")
					continue
				}
			}

			fallthrough
		case "avg pool", "max pool":
		RemakePool:


			args := new(operators.PoolArgs)

			printf("Please enter preferred input dimensions: ")
		PoolInputDims:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.InputDims = make([]int, len(dimStrs))
				inputSize := 1
				for i := range dimStrs {
					if args.InputDims[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue PoolInputDims
					} else if args.InputDims[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue PoolInputDims
					}

					inputSize *= args.InputDims[i]
				}

				if numInputs != inputSize {
					printf("Product of input dimensions (%d) is not equal to number of inputs (%d). Try again or 'abort'.\n", inputSize, numInputs)
					continue
				}

				break
			}

			printf("Please enter preferred output dimensions: ")
		PoolDims:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Dims = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Dims[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue PoolDims
					} else if args.Dims[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue PoolDims
					}
				}

				if len(args.Dims) != len(args.InputDims) {
					printf("Number of output and input dimensions should match (%d != %d)\n", len(args.Dims), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto RemakePool
						} else if sc.Text() == "n" {
							printf("Please enter preferred output dimensions: ")
							continue PoolDims
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred filter size (for each dimension): ")
		PoolFilter:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Filter = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Filter[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue PoolFilter
					} else if args.Filter[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue PoolFilter
					}
				}

				if len(args.Filter) != len(args.InputDims) {
					printf("Number of filter and input dimensions should match (%d != %d)\n", len(args.Filter), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto RemakePool
						} else if sc.Text() == "n" {
							printf("Please enter preferred filter dimensions: ")
							continue PoolFilter
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred stride (for each dimension): ")
		PoolStride:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Stride = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Stride[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue PoolStride
					} else if args.Stride[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue PoolStride
					}
				}

				if len(args.Stride) != len(args.InputDims) {
					printf("Number of stride and input dimensions should match (%d != %d)\n", len(args.Stride), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto RemakePool
						} else if sc.Text() == "n" {
							printf("Please enter preferred stride dimensions: ")
							continue PoolStride
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
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

			printf("Please enter preferred input dimensions: ")
		ConvInputDims:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.InputDims = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.InputDims[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue ConvInputDims
					} else if args.InputDims[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue ConvInputDims
					}
				}

				break
			}

			printf("Please enter preferred output dimensions (not including depth): ")
		ConvDims:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Dims = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Dims[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue ConvDims
					} else if args.Dims[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue ConvDims
					}
				}

				if len(args.Dims) != len(args.InputDims) {
					printf("Number of output and input dimensions should match (%d != %d)\n", len(args.Dims), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto GetOperator
						} else if sc.Text() == "n" {
							printf("Please enter preferred output dimensions: ")
							continue ConvDims
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred filter size (for each dimension): ")
		ConvFilter:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Filter = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Filter[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue ConvFilter
					} else if args.Filter[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue ConvFilter
					}
				}

				if len(args.Filter) != len(args.InputDims) {
					printf("Number of filter and input dimensions should match (%d != %d)\n", len(args.Filter), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto GetOperator
						} else if sc.Text() == "n" {
							printf("Please enter preferred filter dimensions: ")
							continue ConvFilter
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred stride (for each dimension): ")
		ConvStride:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.Stride = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.Stride[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue ConvStride
					} else if args.Stride[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue ConvStride
					}
				}

				if len(args.Stride) != len(args.InputDims) {
					printf("Number of stride and input dimensions should match (%d != %d)\n", len(args.Stride), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto GetOperator
						} else if sc.Text() == "n" {
							printf("Please enter preferred stride dimensions: ")
							continue ConvStride
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred zero padding (for each dimension): ")
		ConvPadding:
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				dimStrs := strings.Split(sc.Text(), " ")
				if len(dimStrs) == 0 {
					println("There should be at least one dimension given.")
					continue
				}

				args.ZeroPadding = make([]int, len(dimStrs))
				for i := range dimStrs {
					if args.ZeroPadding[i], err = strconv.Atoi(dimStrs[i]); err != nil {
						println("Dimensions given should be integers. Try again.")
						continue ConvPadding
					} else if args.ZeroPadding[i] < 1 {
						println("Dimensions should be ≥ 1. Try again.")
						continue ConvPadding
					}
				}

				if len(args.ZeroPadding) != len(args.InputDims) {
					printf("Number of zero-padding and input dimensions should match (%d != %d)\n", len(args.ZeroPadding), len(args.InputDims))
					printf("Restart construction from input dimensions? (y/n): ")
					for {
						if !sc.Scan() {
							return errors.Errorf("Scanner.Scan() failed")
						}

						if sc.Text() == "y" {
							goto GetOperator
						} else if sc.Text() == "n" {
							printf("Please enter preferred zero-padding dimensions: ")
							continue ConvPadding
						} else {
							printf("Please enter 'y' or 'n': ")
							continue
						}
					}
				}

				break
			}

			printf("Please enter preferred depth: ")
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				if args.Depth, err = strconv.Atoi(sc.Text()); err != nil {
					println("Please enter a number for depth. (1 if unsure).")
					continue
				} else if args.Depth < 1 {
					println("Depth should be ≥ 1. Try again.")
					continue
				}

				break
			}

			printf("Please enter whether or not to have biases (y/n): ")
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "y" {
					args.Biases = true
				} else if sc.Text() == "n" {
					args.Biases = false
				} else {
					printf("Please enter 'y' or 'n': ")
					continue
				}
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
			for {
				if !sc.Scan() {
					return errors.Errorf("Scanner.Scan() failed")
				}

				if sc.Text() == "abort" {
					println("Aborting.")
					return nil
				}

				if size, err = strconv.Atoi(sc.Text()); err != nil {
					printf("Please enter a number for output size of layer: ")
					continue
				} else if size < 1 {
					printf("Size should be ≥ 1. Try again: ")
					continue
				}

				break
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
				printf("Can't make layer that contains a double-quote. Enter a name for this layer: ")
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

	// returns true if it was not aborted
	finish := func() (bool, error) {
		println("Please list which layers should be outputs:")
		for {

			if !sc.Scan() {
				return false, errors.Errorf("Scanner.Scan() failed while trying to finish")
			}

			if sc.Text() == "abort" {
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
			if aborted, err := finish(); !aborted {
				if err == nil {
					return net, types, nil
				}

				return nil, "", errors.Wrapf(err, "")
			}
		case "abort":
			println("Aborting.")
			return nil, "", errors.Errorf("Aborted")
		case "help":
			help()
		default:
			printf("Unknonwn command. ")
		}
	}
}
