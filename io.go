package badstudent

import (
	"bufio"
	"github.com/pkg/errors"
	"os"
	"strconv"
	"strings"
	// "fmt"
)

// main_file should not be a number
const main_file string = "main"

func (net *Network) printMain(dirPath string) error {
	f, err := os.Create(dirPath + "/" + main_file + ".txt")
	if err != nil {
		return errors.Wrapf(err, "Can't print main network to file, couldn't create file %s in %s\n", main_file, dirPath)
	}

	defer f.Close()

	// print the number of nodes
	if _, err = f.WriteString(strconv.Itoa(len(net.nodesByID)) + "\n"); err != nil {
		return err // should be changed eventually
	}

	// print the id's of each input node, in order, separated by spaces
	str := ""
	for i, in := range net.inputs.nodes {
		if i > 0 {
			str += " "
		}

		str += strconv.Itoa(in.id)
	}
	if _, err = f.WriteString(str + "\n"); err != nil {
		return err // should be changed eventually
	}

	// print the id's of each output node, in order, separated by spaces
	str = ""
	for i, out := range net.outputs.nodes {
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

func (n *Node) printNode(dirPath string) error {
	f, err := os.Create(dirPath + "/" + strconv.Itoa(n.id) + ".txt")
	if err != nil {
		return err // should be more descriptive
	}

	defer f.Close()

	// print id
	f.WriteString(strconv.Itoa(n.id) + "\n")

	// print name
	f.WriteString(n.Name + "\n")

	// print size
	f.WriteString(strconv.Itoa(n.Size()) + "\n")

	// print list of inputs by id
	str := ""
	for i, in := range n.inputs.nodes {
		if i != 0 {
			str += " "
		}

		str += strconv.Itoa(in.id)
	}
	str += "\n"
	f.WriteString(str)

	return nil
}

// saves the network to the specified path, creating a directory to contain it (with permissions 0700)
//
// The provided path should not have all directories already created, unless overwrite is 'true'
//
// if 'overwrite' is false and the directory already exists, Save will return error.
func (net *Network) Save(dirPath string, overwrite bool) error {
	var err error

	// check if the folder already exists
	if _, err = os.Stat(dirPath); err == nil {
		if !overwrite {
			return errors.Errorf("Can't save network, folder already exists, and overwrite is not enabled")
		}

		if err = os.RemoveAll(dirPath); err != nil {
			return errors.Errorf("Can't save network, couldn't remove pre-existing folder to overwrite")
		}
	}

	if err = os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Wrapf(err, "Couldn't make directory to save network")
	}

	net.printMain(dirPath)

	for _, n := range net.nodesByID {
		if err = n.printNode(dirPath); err != nil {
			return err // should be more descriptive
		}

		if err = n.typ.Save(n, dirPath+"/"+strconv.Itoa(n.id)); err != nil {
			return err // should be more descriptive
		}
	}

	return nil
}

// Loads the network from a version previously saved in a directory
//
// The provided path should be to the containing folder, the same as it would have been to Save() the network
// 'types' and 'aux' should be maps of name of Node to their values
// 'aux' is used to provide other information that may be necessary for certain constructors
//
// when finished, Load calls *Network.SetOutputs to finalize the network
func Load(dirPath string, types map[string]Operator, aux map[string][]interface{}) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(dirPath); err != nil {
		return nil, errors.Errorf("Can't load network, containing directory does not exist")
	}

	main, err := os.Open(dirPath + "/" + main_file + ".txt")
	if err != nil {
		return nil, errors.Errorf("Can't load network, main file does not exist")
	}
	defer main.Close()

	formatErr := errors.Errorf("Can't load network, main file is incompatible")
	sc := bufio.NewScanner(main)

	net := new(Network)
	net.nodesByName = make(map[string]*Node)

	// make() net.nodesByID with correct length
	{
		if !sc.Scan() {
			return nil, formatErr
		}

		var numNodes int
		if numNodes, err = strconv.Atoi(sc.Text()); err != nil {
			return nil, formatErr
		}

		net.nodesByID = make([]*Node, numNodes)
	}

	// get list of input and output nodes
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

	// make all of the nodes in the network, in order by id
	for id := range net.nodesByID {
		if err = net.remakeNode(dirPath, id); err != nil {
			return nil, errors.Wrapf(err, "Can't load network: failed to load node (id: %d)\n", id)
		}

		subDir := dirPath + "/" + strconv.Itoa(id)
		n := net.nodesByID[id]

		if types[n.Name] == nil {
			return nil, errors.Errorf("Can't load network, no given Operator for node %v", n)
		}
		n.typ = types[n.Name]

		if err = types[n.Name].Load(n, subDir, aux[n.Name]); err != nil {
			return nil, errors.Wrapf(err, "Can't load network, failed to load Operator for node %v (id: %d)\n", n, id)
		}
	}

	// check that the inputs to the network are the same as what has been provided
	// -- essentially checking that everything adds up
	for i, id := range inputsByID {
		if net.inputs.nodes[i] != net.nodesByID[id] {
			return nil, errors.Errorf("Network input %d (%v) does not match supposed network input (from %s.txt) (%v)", i, net.inputs.nodes[i], main_file, net.nodesByID[id])
		}
	}

	// set the outputs to the network
	{
		outputs := make([]*Node, len(outputsByID))
		for i, id := range outputsByID {
			outputs[i] = net.nodesByID[id]
		}

		if err := net.SetOutputs(outputs...); err != nil {
			return nil, errors.Wrapf(err, "Loaded network; could not set outputs\n")
		}
	}

	return net, nil
}

// 'dirPath' should be the same path for the loading of the network
// -- it should not be the path to a file
func (net *Network) remakeNode(dirPath string, id int) error {
	f, err := os.Open(dirPath + "/" + strconv.Itoa(id) + ".txt")
	if err != nil {
		return errors.Errorf("Can't load network, file for node #%d doesn't exist", id)
	}

	defer f.Close()

	n := new(Node)
	n.host = net

	sc := bufio.NewScanner(f)
	formatErr := errors.Errorf("Can't load network, file for node #%d has wrong format", id)

	// set node id - check that the id of the file matches the id of its name
	{
		if !sc.Scan() {
			return formatErr
		}

		if n.id, err = strconv.Atoi(sc.Text()); err != nil {
			return formatErr
		}

		if n.id != id {
			return errors.Errorf("Can't load network, mismatch between file name and content of node %d", id)
		}
	}

	// set node name, size
	{
		if !sc.Scan() {
			return formatErr
		}
		n.Name = sc.Text()
		net.nodesByName[n.Name] = n

		if !sc.Scan() {
			return formatErr
		}
		size, err := strconv.Atoi(sc.Text())
		if err != nil {
			return formatErr
		}
		n.values = make([]float64, size)
		n.deltas = make([]float64, size)
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
			net.inputs.add(n)
		} else {
			n.inputs = new(nodeGroup)
			for i := range ids {
				in := net.nodesByID[ids[i]]
				n.inputs.add(in)

				in.outputs.add(n)
			}
		}
	}

	net.nodesByID[id] = n

	return nil
}
