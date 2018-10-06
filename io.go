package badstudent

import (
	"encoding/json"
	"github.com/pkg/errors"
	"os"
	"strconv"
)

var ops map[string]func() Operator
var opts map[string]func() Optimizer

func init() {
	ops = make(map[string]func() Operator)
	opts = make(map[string]func() Optimizer)
}

// RegisterOperator updates internals so that Load() will recognize the
// Operator. Returns error if `name` is already present.
func RegisterOperator(name string, f func() Operator) error {
	if _, ok := ops[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	}

	ops[name] = f
	return nil
}

// RegisterOptimizer updates internals so that Load() will recognize the
// Optimizer. Returns error if `name` is already present.
func RegisterOptimizer(name string, f func() Optimizer) error {
	if _, ok := opts[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	}

	opts[name] = f
	return nil
}

type proxy_Network struct {
	// the IDs of each of the Nodes in the outputs
	OutputsID []int
	NumNodes  int
}

type proxy_Node struct {
	Size       int
	Name       string
	TypeString string
	InputsID   []int
	Delay      int
}

func nodesToIDs(nodes []*Node) []int {
	ids := make([]int, len(nodes))
	for i := range nodes {
		ids[i] = nodes[i].id
	}
	return ids
}

func idsToNodes(nodesByID []*Node, ids []int) []*Node {
	nodes := make([]*Node, len(ids))
	for i := range ids {
		nodes[i] = nodesByID[ids[i]]
	}
	return nodes
}

const (
	main_file string = "main.net"
	op_ext    string = "op"
	opt_ext   string = "opt"
)

func (net *Network) writeFile(dirPath string) error {
	f, err := os.Create(dirPath + "/" + main_file)
	if err != nil {
		return errors.Wrapf(err, "Failed to create file %s in %s\n", main_file, dirPath)
	}

	defer f.Close()

	// convert the network to a format that can be written
	var proxy proxy_Network
	{
		proxy = proxy_Network{
			OutputsID: nodesToIDs(net.outputs.nodes),
			NumNodes:  len(net.nodesByID),
		}
	}

	enc := json.NewEncoder(f)
	if err = enc.Encode(proxy); err != nil {
		return errors.Wrapf(err, "Failed to encode Network proxy\n")
	}

	return nil
}

func (n *Node) writeFile(dirPath string) error {
	f, err := os.Create(dirPath + "/" + strconv.Itoa(n.id) + ".node")
	if err != nil {
		return errors.Wrapf(err, "Failed to create a save file for Node %v (file %q in %s)\n", n, strconv.Itoa(n.id)+".node", dirPath)
	}

	var ts string
	if !n.IsInput() {
		ts = n.typ.TypeString()
	}

	// convert the node to a format that can be written
	proxy := proxy_Node{
		Size:       n.Size(),
		Name:       n.Name(),
		TypeString: ts,
		InputsID:   nodesToIDs(n.inputs.nodes),
		Delay:      n.Delay(),
	}

	enc := json.NewEncoder(f)
	if err = enc.Encode(proxy); err != nil {
		f.Close()
		return errors.Wrapf(err, "Failed to encode prody for Node %v (id: %d)\n", n, n.id)
	}

	f.Close()

	// if n is an input, it won't have an operator
	if !n.IsInput() {
		if err = n.typ.Save(n, dirPath+"/"+strconv.Itoa(n.id)+"/"+op_ext); err != nil {
			return errors.Wrapf(err, "Failed to save Operator for Node %v (id: %d)\n", n)
		}
	}

	return nil
}

// Saves the Network, creating a directory (with permissions 0700) as the given path to contain it.
//
// If overwrite is false, it will not overwrite any pre-existing directories - including the path given -
// and will return error.
func (net *Network) Save(dirPath string, overwrite bool) error {
	// check if the folder already exists
	if _, err := os.Stat(dirPath); err == nil {
		if !overwrite {
			return errors.Errorf("Directory %s already exists, and overwrite is not enabled", dirPath)
		}

		if err := os.RemoveAll(dirPath); err != nil {
			return errors.Errorf("Failed to remove pre-existing folder to overwrite")
		}
	}

	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return errors.Wrapf(err, "Failed to make save directory\n")
	}

	if err := net.writeFile(dirPath); err != nil {
		return errors.Wrapf(err, "Failed to save network overview\n")
	}

	for _, n := range net.nodesByID {
		if err := n.writeFile(dirPath); err != nil {
			return errors.Wrapf(err, "Failed to save Node %v\n", n)
		}
	}

	return nil
}

// Loads the network from a version previously saved in a directory
//
// The provided path should be to the containing folder, the same as it would have been to save the network
// 'types' and 'aux' should be maps of name of Node to their Operator (for types) and any extra information
// necessary to reconstruct that Operator (aux)
//
// aux will be provided to Operator.Load()
func Load(dirPath string) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(dirPath); err != nil {
		return nil, errors.Errorf("Containing directory does not exist")
	}

	main, err := os.Open(dirPath + "/" + main_file)
	if err != nil {
		return nil, errors.Errorf("Overview file does not exist")
	}

	net_proxy := new(proxy_Network)
	{
		dec := json.NewDecoder(main)
		if err := dec.Decode(net_proxy); err != nil {
			return nil, errors.Wrapf(err, "Error encountered while decoding overview file\n")
		}
		main.Close()
	}

	net := new(Network)

	typs := make([]Operator, net_proxy.NumNodes)
	ins := make([][]int, net_proxy.NumNodes)
	delays := make([]int, net_proxy.NumNodes)

	for id := 0; id < net_proxy.NumNodes; id++ {

		f, err := os.Open(dirPath + "/" + strconv.Itoa(id) + ".node")
		if err != nil {
			return nil, errors.Wrapf(err, "File for node #%d does not exist\n", id)
		}

		// proxy node
		pn := new(proxy_Node)
		{
			dec := json.NewDecoder(f)
			err = dec.Decode(pn)
			f.Close()
			if err != nil {
				return nil, errors.Wrapf(err, "Error encountered while decoding file for Node %q (id %d)\n", pn.Name, id)
			}
		}

		_, err = net.Placeholder(pn.Name, pn.Size)
		if err != nil {
			return nil, errors.Wrapf(err, "Failed to add placeholder Node %q (id %d) to reconstructing Network\n", pn.Name, id)
		}

		ins[id] = pn.InputsID
		delays[id] = pn.Delay

		if len(ins[id]) != 0 {
			t, ok := ops[pn.TypeString]
			if !ok {
				return nil, errors.Errorf("Unknown operator %q has not been registered", pn.TypeString)
			}

			typs[id] = t()
			if err = typs[id].Load(dirPath + "/" + strconv.Itoa(id) + "/" + op_ext); err != nil {
				return nil, errors.Wrapf(err, "Failed to load Operator for Node %q (id %d)\n", pn.Name, id)
			}
		}
	}

	for id, n := range net.nodesByID {
		inputs := idsToNodes(net.nodesByID, ins[id])

		n.Replace(typs[id], inputs...)

		if delays[id] != 0 {
			n.SetDelay(delays[id])
		}
	}

	// set the outputs to the network
	if err := net.SetOutputs(idsToNodes(net.nodesByID, net_proxy.OutputsID)...); err != nil {
		return nil, errors.Wrapf(err, "Could not set outputs\n")
	}

	return net, nil
}
