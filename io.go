package badstudent

// io currently does not support saving or loading hyperparameters

import (
	"encoding/json"
	"github.com/pkg/errors"
	"os"
	"strconv"
)

// Go 2.0, can you arrive any sooner?
var ops map[string]func() Operator
var opts map[string]func() Optimizer
var cfs map[string]func() CostFunction
var hps map[string]func() HyperParameter

func init() {
	ops = make(map[string]func() Operator)
	opts = make(map[string]func() Optimizer)
	cfs = make(map[string]func() CostFunction)
	hps = make(map[string]func() HyperParameter)
}

// RegisterOperator updates internals so that Load() will recognize the Operator.
// Returns error if `name` is already present or if the provided Operator is
// invalid: i.e. is neither Elementwise nor Layer.
func RegisterOperator(name string, f func() Operator) error {
	if _, ok := ops[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	} else if !isValid(f()) {
		return errors.Errorf("Function does not provide valid operators (neither Layer nor Elementwise)")
	}

	ops[name] = f
	return nil
}

// RegisterOptimizer updates internals so that Load() will recognize the Optimizer.
// Returns error if `name` is already present.
func RegisterOptimizer(name string, f func() Optimizer) error {
	if _, ok := opts[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	}

	opts[name] = f
	return nil
}

// RegisterCostFunction updates internals so that Load() will recognize the
// CostFunction. Returns error if `name` is already present.
func RegisterCostFunction(name string, f func() CostFunction) error {
	if _, ok := cfs[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	}

	cfs[name] = f
	return nil
}

// RegisterHyperParameter updates internals so that Load() will recognize the
// HyperParameter. Returns error if `name` is already present.
func RegisterHyperParameter(name string, f func() HyperParameter) error {
	if _, ok := hps[name]; ok {
		return errors.Errorf("Name %s has already been registered", name)
	}

	hps[name] = f
	return nil
}

type proxy_Network struct {
	// the IDs of each of the Nodes in the outputs
	OutputsID []int
	NumNodes  int
	CFType    string // the type-string of the cost function
}

type proxy_Node struct {
	Size       int
	Name       string
	TypeString string
	OptString  string            // usually nil
	HPTypes    map[string]string // map of names to type-strings
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
	typ_ext   string = "typ"
	opt_ext   string = "opt"
	hp_pref   string = "hp_"
)

func (net *Network) writeFile(dirPath string) error {
	f, err := os.Create(dirPath + "/" + main_file)
	if err != nil {
		return errors.Wrapf(err, "Failed to create file %q in %q\n", main_file, dirPath)
	}

	defer f.Close()

	p := proxy_Network{
		OutputsID: nodesToIDs(net.outputs.nodes),
		NumNodes:  len(net.nodesByID),
		CFType:    net.cf.TypeString(),
	}

	enc := json.NewEncoder(f)
	if err = enc.Encode(p); err != nil {
		return errors.Wrapf(err, "Failed to write information to file %q in %q\n", main_file, dirPath)
	}

	return nil
}

func (n *Node) writeFile(dirPath string) error {
	path := dirPath + "/" + strconv.Itoa(n.id)

	var typStr, optStr string
	if n.typ != nil {
		typStr = n.typ.TypeString()
	}
	if n.opt != nil {
		optStr = n.opt.TypeString()
	}

	// map of name to type
	hpTypes := make(map[string]string)
	for name, hp := range n.hyperParams {
		hpTypes[name] = hp.TypeString()
	}

	p := proxy_Node{
		Size:       n.Size(),
		Name:       n.Name(),
		TypeString: typStr,
		OptString:  optStr,
		HPTypes:    hpTypes,
		InputsID:   nodesToIDs(n.inputs.nodes),
		Delay:      n.Delay(),
	}

	f, err := os.Create(path + ".node")
	if err != nil {
		return errors.Errorf("Failed to create save file (file %q in %q)", strconv.Itoa(n.id)+".node", dirPath)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	err = enc.Encode(p)

	if err != nil {
		return errors.Errorf("Failed to write information (file %q in %q)", strconv.Itoa(n.id)+".node", dirPath)
	}

	if n.typ != nil {
		if st, ok := n.typ.(Storable); ok {
			if err = st.Save(n, path+"/"+typ_ext); err != nil {
				return errors.Wrapf(err, "Failed to save Operator\n")
			}
		}
	}

	if n.opt != nil {
		if st, ok := n.opt.(Storable); ok {
			if err = st.Save(n, path+"/"+opt_ext); err != nil {
				return errors.Wrapf(err, "Failed to save Optimizer\n")
			}
		}
	}

	for name, hp := range n.hyperParams {
		if st, ok := hp.(Storable); ok {
			if err = st.Save(n, path+"/"+hp_pref+name); err != nil {
				return errors.Wrapf(err, "Failed to save HyperParameter (name: %s, type: %s)\n", name, hp.TypeString())
			}
		}
	}

	return nil
}

// Save saves the Network to a given directory. overwrite indicates whether or not
// to remove any other files that may already exist there to replace them. The
// directory should not already exist unless overwrite is true.
//
// Save does not error cleanly. If it stops midway through writing the Network, it
// will leave the files unfinished.
func (net *Network) Save(dirPath string, overwrite bool) error {
	// check if the folder already exists
	if _, err := os.Stat(dirPath); err == nil {
		if !overwrite {
			return errors.Errorf("Directory %s already exists, overwrite=false", dirPath)
		}

		if err := os.RemoveAll(dirPath); err != nil {
			return errors.Errorf("Failed to remove pre-existing folder to overwrite")
		}
	}

	// '0700' indicates universal read/write permissions
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

func Load(dirPath string) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(dirPath); err != nil {
		return nil, errors.Errorf("Containing directory does not exist")
	}

	// Load the main network
	var pNet proxy_Network
	{
		f, err := os.Open(dirPath + "/" + main_file)
		if err != nil {
			return nil, errors.Errorf("Failed to open Network overview (file %q in %q)", main_file, dirPath)
		}

		defer f.Close()

		dec := json.NewDecoder(f)
		if err = dec.Decode(&pNet); err != nil {
			return nil, errors.Errorf("Failed to parse Network overview (file %q in %q)", main_file, dirPath)
		}
	}

	net := new(Network)
	pNodes := make([]proxy_Node, pNet.NumNodes)

	// Load the Nodes, make placeholders
	for id := 0; id < pNet.NumNodes; id++ {
		path := dirPath + "/" + strconv.Itoa(id)
		f, err := os.Open(path + ".node")
		if err != nil {
			return nil, errors.Errorf("Failed to open save for Node (id %d) (file %q in %q)",
				id, strconv.Itoa(id)+".node", dirPath)
		}

		if err = func() error {
			defer f.Close()

			dec := json.NewDecoder(f)
			if err = dec.Decode(&pNodes[id]); err != nil {
				return errors.Errorf("Failed to read save for Node (id %d) (file %q in %q)",
					id, strconv.Itoa(id)+".node", dirPath)
			}

			return nil
		}(); err != nil {
			return nil, err
		}

		net.Placeholder(pNodes[id].Name, pNodes[id].Size)
		if net.err != nil {
			return nil, errors.Wrapf(net.err, "Failed to add placeholder for Node %q (id %d) to reconstruct Network\n",
				pNodes[id].Name, id)
		}
	}

	// replace placeholders
	for id, n := range net.nodesByID {
		inputs := idsToNodes(net.nodesByID, pNodes[id].InputsID)

		var typ Operator
		var opt Optimizer
		if len(inputs) > 0 {
			if ops[pNodes[id].TypeString] == nil {
				return nil, errors.Errorf("Operator type %q has not been registered", pNodes[id].TypeString)
			}

			typ = ops[pNodes[id].TypeString]()
			if st, ok := typ.(Storable); ok {
				if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + typ_ext); err != nil {
					return nil, errors.Wrapf(err, "Failed to Load Operator for Node %q (id %d)\n", pNodes[id].Name, id)
				}
			}

			if _, ok := typ.(Adjustable); ok {
				if opts[pNodes[id].OptString] == nil {
					return nil, errors.Errorf("Optimizer type %q has not been registered", pNodes[id].OptString)
				}

				opt = opts[pNodes[id].OptString]()

				if st, ok := opt.(Storable); ok {
					if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + opt_ext); err != nil {
						return nil, errors.Wrapf(err, "Failed to Load Optimizer for Node %q (id %d)\n", pNodes[id].Name, id)
					}
				}
			}
		}

		n.Replace(typ, inputs...)
		if net.err != nil {
			return nil, errors.Wrapf(net.err, "Failed to replace placeholder for Node %q (id %d) to reconstruct Network\n",
				pNodes[id].Name, id)
		}

		if opt != nil {
			n.Opt(opt)
		}
		if pNodes[id].Delay != 0 {
			n.SetDelay(pNodes[id].Delay)
		}

		for name, ts := range pNodes[id].HPTypes {
			if hps[ts] == nil {
				return nil, errors.Errorf("HyperParameter type %q has not been registered", ts)
			}

			hp := hps[ts]()

			if st, ok := hp.(Storable); ok {
				if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + hp_pref + name); err != nil {
					return nil, errors.Wrapf(err, "Failed to load HyperParameter %q for Node %q (id %d)\n", ts, n.name, id)
				}
			}

			n.AddHP(name, hp)
			if net.err != nil {
				return nil, errors.Wrapf(net.err, "Failed to add HyperParameter %q to Node %q (id %d)\n", ts, n.name, id)
			}
		}
	}

	if cfs[pNet.CFType] == nil {
		return nil, errors.Errorf("CostFunction type %q has not been registered")
	}

	cf := cfs[pNet.CFType]()

	if err := net.finalize(true, cf, idsToNodes(net.nodesByID, pNet.OutputsID)...); err != nil {
		return nil, errors.Wrapf(err, "Failed to finalize Network\n")
	}

	return net, nil
}
