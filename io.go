package badstudent

import (
	"github.com/pkg/errors"
	"os"
	"bufio"
	"strings"
	"strconv"
	// "fmt"
)

// main_file should not be a number
const main_file string = "main"

func (net *Network) printMain(dirPath string) error {
	f, err := os.Create(dirPath + "/" + main_file)
	if err != nil {
		return errors.Wrapf(err, "Can't print main network to file, couldn't create file %s in %s\n", main_file, dirPath)
	}

	defer f.Close()

	// print the number of layers
	if _, err = f.WriteString(strconv.Itoa(len(net.layersByID)) + "\n"); err != nil {
		return err // should be changed eventually
	}

	// print the id's of each input layer, in order, separated by spaces
	str := ""
	for i, in := range net.inLayers {
		if i > 0 {
			str += " "
		}

		str += strconv.Itoa(in.id)
	}
	if _, err = f.WriteString(str + "\n"); err != nil {
		return err // should be changed eventually
	}

	// print the id's of each output layer, in order, separated by spaces
	str = ""
	for i, out := range net.outLayers {
		if i > 0 {
			str += " "
		}

		str += strconv.Itoa(out.id)
	}
	if _, err = f.WriteString(str + "\n"); err != nil {
		return err // should be changed eventually
	}

	return nil
}

func (l *Layer) printLayer(dirPath string) error {
	f, err := os.Create(dirPath + "/" + strconv.Itoa(l.id) + ".txt")
	if err != nil {
		return err // should be more descriptive
	}

	defer f.Close()

	// print id
	f.WriteString(strconv.Itoa(l.id) + "\n")

	// print name
	f.WriteString(l.Name + "\n")

	// print size
	f.WriteString(strconv.Itoa(l.Size()) + "\n")

	// print dimensions
	str := ""
	for i := range l.dims {
		if i != 0 {
			str += " "
		}

		str += strconv.Itoa(l.dims[i])
	}
	str += "\n"
	f.WriteString(str)

	// print list of inputs by id
	str = ""
	for i := range l.inputs {
		if i != 0 {
			str += " "
		}

		str += strconv.Itoa(l.inputs[i].id)
	}
	str += "\n"
	f.WriteString(str)

	// print number of outputs
	f.WriteString(strconv.Itoa(len(l.outputs)) + "\n")

	return nil
}

func (net *Network) Save(dirPath string) error {
	var err error

	// check if the folder already exists
	if _, err = os.Stat(dirPath); err == nil {
		return errors.Errorf("Can't save network, folder already exists")
	}

	if err = os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Wrapf(err, "Couldn't make directory to save network")
	}

	net.printMain(dirPath)

	for _, l := range net.layersByID {
		if err = l.printLayer(dirPath); err != nil {
			return err // should be more descriptive
		}

		if err = l.typ.Save(l, dirPath + "/" + strconv.Itoa(l.id)); err != nil {
			return err // should be more descriptive
		}
	}

	return nil
}

// 'dirPath' should be the path to the containing folder, including the name
// 'types' and 'aux' should be maps of name of Layer to Operator | []interface{}.
// this is why each layer is required to be named differently
//
// when finished loading, calls SetOutputs() to finalize the network
func Load(dirPath string, types map[string]Operator, aux map[string][]interface{}) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(dirPath); err != nil {
		return nil, errors.Errorf("Can't load network, containing directory does not exist")
	}

	main, err := os.Open(dirPath + "/" + main_file)
	if err != nil {
		return nil, errors.Errorf("Can't load network, main file does not exist")
	}
	defer main.Close()

	formatErr := errors.Errorf("Can't load network, main file is incompatible")
	sc := bufio.NewScanner(main)
	
	net := new(Network)
	net.layersByName = make(map[string]*Layer)

	// make() net.layersByID with correct length
	{
		if !sc.Scan() {
			return nil, formatErr
		}

		var numLayers int
		if numLayers, err = strconv.Atoi(sc.Text()); err != nil {
			return nil, formatErr
		}

		net.layersByID = make([]*Layer, numLayers)
	}

	// get list of input and output layers
	var inputsByID, outputsByID []int
	{	
		if !sc.Scan() {
			return nil, formatErr
		}

		inStrs := strings.Split(sc.Text(), " ")
		inputsByID = make([]int, len(inStrs))
		for i, str := range inStrs {
			if inputsByID[i], err = strconv.Atoi(str); err != nil {
				return nil, formatErr
			}
		}

		if !sc.Scan() {
			return nil, formatErr
		}

		outStrs := strings.Split(sc.Text(), " ")
		outputsByID = make([]int, len(outStrs))
		for i, str := range outStrs {
			if outputsByID[i], err = strconv.Atoi(str); err != nil {
				return nil, formatErr
			}
		}
	}

	// make all of the layers in the network, in order by id
	for id := range net.layersByID {
		if err = net.remakeLayer(dirPath, id); err != nil {
			return nil, errors.Wrapf(err, "Can't load network: failed to load layer (id: %d)\n", id)
		}

		subDir := dirPath + "/" + strconv.Itoa(id)
		l := net.layersByID[id]

		if types[l.Name] == nil {
			return nil, errors.Errorf("Can't load network, no given Operator for layer %v", l)
		}
		l.typ = types[l.Name]

		if err = types[l.Name].Load(l, subDir, aux[l.Name]); err != nil {
			return nil, errors.Wrapf(err, "Can't load network, failed to load Operator for layer %v (id: %d)\n", l, id)
		}
	}

	// check that the inputs to the network are the same as what has been provided
	// -- essentially checking that everything adds up
	for i, id := range inputsByID {
		if net.inLayers[i] != net.layersByID[id] {
			return nil, errors.Errorf("Network input %d (%v) does not match supposed network input (from %s.txt) (%v)", i, net.inLayers[i], main_file, net.layersByID[id])
		}
	}

	// set the outputs to the network
	{
		outputs := make([]*Layer, len(outputsByID))
		for i, id := range outputsByID {
			outputs[i] = net.layersByID[id]
		}

		if err := net.SetOutputs(outputs...); err != nil {
			return nil, errors.Wrapf(err, "Loaded network; could not set outputs\n")
		}
	}

	return net, nil
}

// 'dirPath' should be the same path for the loading of the network
// -- it should not be the path to a file
func (net *Network) remakeLayer(dirPath string, id int) error {
	f, err := os.Open(dirPath + "/" + strconv.Itoa(id) + ".txt")
	if err != nil {
		return errors.Errorf("Can't load network, file for layer #%d doesn't exist", id)
	}

	defer f.Close()
	
	l := new(Layer)
	l.hostNetwork = net
	l.status = initialized

	sc := bufio.NewScanner(f)
	formatErr := errors.Errorf("Can't load network, file for layer #%d has wrong format", id)

	// set layer id - check that the id of the file matches the id of its name
	{
		if !sc.Scan() {
			return formatErr
		}

		if l.id, err = strconv.Atoi(sc.Text()); err != nil {
			return formatErr
		}

		if l.id != id {
			return errors.Errorf("Can't load network, mismatch between file name and content of layer %d", id)
		}
	}

	// set layer name, size, dimensions
	{
		if !sc.Scan() {
			return formatErr
		}
		l.Name = sc.Text()
		net.layersByName[l.Name] = l
	
		if !sc.Scan() {
			return formatErr
		}
		size, err := strconv.Atoi(sc.Text())
		if err != nil {
			return formatErr
		}
		l.values = make([]float64, size)
		l.deltas = make([]float64, size)
	
		if !sc.Scan() {
			return formatErr
		}

		if sc.Text() != "" {
			strs := strings.Split(sc.Text(), " ")
			l.dims = make([]int, len(strs))
			for i, str := range strs {
				if l.dims[i], err = strconv.Atoi(str); err != nil {
					return formatErr
				}
			}
		}
	}

	// set the inputs to l
	{
		if !sc.Scan() {
			return formatErr
		}

		var ids []int
		if sc.Text() != "" {
			strs := strings.Split(sc.Text(), " ")
			ids = make([]int, len(strs))
			for i, str := range strs {
				if ids[i], err = strconv.Atoi(str); err != nil {
					return formatErr
				}
			}
		}

		if len(ids) == 0 {
			net.inLayers = append(net.inLayers, l)
		} else {
			l.inputs = make([]*Layer, len(ids))
			l.numInputs = make([]int, len(ids))
			totalInputs := 0
			for i := range ids {
				in := net.layersByID[ids[i]]
				l.inputs[i] = in

				totalInputs += in.Size()
				l.numInputs[i] = totalInputs

				in.outputs = append(in.outputs, l)
			}
		}
	}

	// set capacity of outputs
	{
		if !sc.Scan() {
			return formatErr
		}

		capacity, err := strconv.Atoi(sc.Text())
		if err != nil {
			return formatErr
		}

		l.outputs = make([]*Layer, 0, capacity)
	}

	net.layersByID[id] = l

	return nil
}
