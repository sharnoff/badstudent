package badstudent

import (
	"encoding/json"
	"github.com/pkg/errors"
	"os"
	"strconv"
)

type proxy_Network struct {
	// the IDs of each of the Nodes in the outputs
	OutputsID []int
	NamesByID []string
}

type proxy_Node struct {
	Size     int
	InputsID []int
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

// referred to in error messages as "overview"
const main_file string = "main.net"

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
		}

		proxy.NamesByID = make([]string, len(net.nodesByID))
		for i := range net.nodesByID {
			proxy.NamesByID[i] = net.nodesByID[i].name
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

	// convert the node to a format that can be written
	proxy := proxy_Node{
		Size:     n.Size(),
		InputsID: nodesToIDs(n.inputs.nodes),
	}

	enc := json.NewEncoder(f)
	if err = enc.Encode(proxy); err != nil {
		f.Close()
		return errors.Wrapf(err, "Failed to encode prody for Node %v (id: %d)\n", n, n.id)
	}

	f.Close()

	if err = n.typ.Save(n, dirPath+"/"+strconv.Itoa(n.id)); err != nil {
		return errors.Wrapf(err, "Failed to save Operator for Node %v (id: %d)\n", n)
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
func Load(dirPath string, types map[string]Operator, aux map[string][]interface{}) (*Network, error) {
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
	net.nodesByName = make(map[string]*Node)
	net.inputs = new(nodeGroup)

	for id := 0; id < len(net_proxy.NamesByID); id++ {
		name := net_proxy.NamesByID[id]

		f, err := os.Open(dirPath + "/" + strconv.Itoa(id) + ".node")
		if err != nil {
			return nil, errors.Wrapf(err, "File for node %q (id %d) does not exist\n", name, id)
		}

		node_proxy := new(proxy_Node)
		{
			dec := json.NewDecoder(f)
			if err := dec.Decode(node_proxy); err != nil {
				return nil, errors.Wrapf(err, "Error encountered while decoding file for Node %q (id %d)\n", name, id)
			}
			f.Close()
		}

		loadType := func(n *Node) (Operator, error) {
			return types[name], types[name].Load(n, dirPath+"/"+strconv.Itoa(id), aux[name])
		}

		if err := net.addNode(name, node_proxy, loadType); err != nil {
			return nil, errors.Wrapf(err, "Failed to add Node to network during reconstruction\n", name, id)
		}
	}

	// set the outputs to the network
	if err := net.SetOutputs(idsToNodes(net.nodesByID, net_proxy.OutputsID)...); err != nil {
		return nil, errors.Wrapf(err, "Could not set outputs\n")
	}

	return net, nil
}

// only error returned would be from loadType()
func (net *Network) addNode(name string, proxy *proxy_Node, loadType func(*Node) (Operator, error)) error {
	n := new(Node)
	n.name = name
	n.host = net
	n.id = len(net.nodesByID)

	n.inputs = new(nodeGroup)
	n.outputs = new(nodeGroup)

	n.inputs.add(idsToNodes(net.nodesByID, proxy.InputsID)...)

	n.values = make([]float64, proxy.Size)
	n.deltas = make([]float64, proxy.Size)

	var err error
	if n.typ, err = loadType(n); err != nil {
		return errors.Wrapf(err, "Failed to load operator for Node %v (id %d)\n", n, n.id)
	}

	if num(n.inputs) == 0 {
		net.inputs.add(n)
	} else {
		for _, in := range n.inputs.nodes {
			in.outputs.add(n)
		}
	}

	net.nodesByName[name] = n
	net.nodesByID = append(net.nodesByID, n)

	return nil
}
