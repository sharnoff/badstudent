package badstudent

import (
	"encoding/json"
	"fmt"
	"github.com/pkg/errors"
	"os"
	"os/exec"
	"strconv"
)

// RegisterNamePresentError documents errors resulting from conflicting names given by
// TypeString() while Registering types
type RegisterNamePresentError struct {
	typ  string
	name string
}

func (err RegisterNamePresentError) Error() string {
	return "Failed to register " + err.typ + ", \"" + err.name + "\" was already present."
}

// Register updates internals so that Load() will recognize the given type. This
// function does not pertain to the average use case; it is only necessary for creating
// custom types. Register is given a function of the form func() <T>, where <T> can be any
// of:
//	(a) Operator
//	(b) Optimizer
//	(c) CostFunction
//	(d) HyperParameter
//	(e) Penalty
//
// Register has several errors that can be returned:
//	(1) NilArgError, if fn is nil;
//	(2) ErrRegisterWrongType, if fn is not one of the types listed above;
//	(3) ErrRegisterNilReturn, if fn returns a nil interface; and
//	(4) RegisterNamePresentError, if fn().TypeString() has already been registered.
func Register(fn interface{}) error {
	if fn == nil {
		return NilArgError{"fn"}
	}

	switch t := fn.(type) {
	case func() Operator:
		r := t()
		if r == nil {
			return ErrRegisterNilReturn 
		} else if ops[r.TypeString()] != nil {
			return RegisterNamePresentError{"Operator", r.TypeString()}
		}

		ops[r.TypeString()] = t
	case func() Optimizer:
		r := t()
		if r == nil {
			return ErrRegisterNilReturn
		} else if opts[r.TypeString()] != nil {
			return RegisterNamePresentError{"Optimizer", r.TypeString()}
		}

		opts[r.TypeString()] = t
	case func() CostFunction:
		r := t()
		if r == nil {
			return ErrRegisterNilReturn
		} else if cfs[r.TypeString()] != nil {
			return RegisterNamePresentError{"CostFunction", r.TypeString()}
		}

		cfs[r.TypeString()] = t
	case func() HyperParameter:
		r := t()
		if r == nil {
			return ErrRegisterNilReturn
		} else if hps[r.TypeString()] != nil {
			return RegisterNamePresentError{"HyperParameter", r.TypeString()}
		}

		hps[r.TypeString()] = t
	case func() Penalty:
		r := t()
		if r == nil {
			return ErrRegisterNilReturn
		} else if pens[r.TypeString()] != nil {
			return RegisterNamePresentError{"Penalty", r.TypeString()}
		}

		pens[r.TypeString()] = t
	default:
		return ErrRegisterWrongType
	}

	return nil
}

// RegisterAll performs Register with all supplied functions. If it encounters an error,
// it returns that error without context. Otherwise, it returns nil.
//
// The simplest way to use RegisterAll can be found in costfuncs/register.go:
//	func init() {
//		list := []interface{}{
//			func() bs.CostFunction { return CrossEntropy() },
//			func() bs.CostFunction { return Huber(0) },
//			func() bs.CostFunction { return MSE() },
//			func() bs.CostFunction { return Abs() },
//		}
//
//		if err := bs.RegisterAll(list); err != nil {
//			panic(err)
//		}
//	}
func RegisterAll(fns []interface{}) error {
	for i := range fns {
		if err := Register(fns[i]); err != nil {
			return err
		}
	}

	return nil
}

// Go 2.0, can you arrive any sooner?
var (
	ops  map[string]func() Operator
	opts map[string]func() Optimizer
	cfs  map[string]func() CostFunction
	hps  map[string]func() HyperParameter
	pens map[string]func() Penalty
)

func init() {
	ops = make(map[string]func() Operator)
	opts = make(map[string]func() Optimizer)
	cfs = make(map[string]func() CostFunction)
	hps = make(map[string]func() HyperParameter)
	pens = make(map[string]func() Penalty)
}

type proxy_Network struct {
	// the IDs of each of the Nodes in the outputs
	OutputsID []int
	NumNodes  int
	Iter      int
	CFType    string // the type-string of the cost function
}

type proxy_Node struct {
	Size       int
	Name       string
	TypeString string
	OptString  string            // -> usually nil
	HPTypes    map[string]string // -> map of names to type-strings
	PenString  string            // -> equals "" if absent
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
	cf_ext    string = "cf"
	typ_ext   string = "typ"
	opt_ext   string = "opt"
	hp_pref   string = "hp_"
	pen_ext   string = "pen"
)

func saveJSON(v interface{}, file string) error {
	f, err := os.Create(file)
	if err != nil {
		return errors.Wrapf(err, "Failed to create file %q\n", file)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(v); err != nil {
		return errors.Wrapf(err, "Failed to write json to file %q\n", file)
	}

	return nil
}

func loadJSON(v interface{}, file string) error {
	f, err := os.Open(file)
	if err != nil {
		return errors.Wrapf(err, "Failed to open file %q\n", file)
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(v); err != nil {
		return errors.Wrapf(err, "Failed to read json from file %q\n", file)
	}

	return nil
}

func (net *Network) writeFile(dirPath string) error {
	p := proxy_Network{
		OutputsID: nodesToIDs(net.outputs.nodes),
		NumNodes:  len(net.nodesByID),
		Iter:      net.iter,
		CFType:    net.cf.TypeString(),
	}

	if err := saveJSON(p, dirPath + "/" + main_file); err != nil {
		return err
	}

	if st, ok := net.cf.(Storable); ok {
		if err := st.Save(dirPath + "/" + cf_ext); err != nil {
			return errors.Wrapf(err, "Failed to save Network CostFunction\n")
		}
	} else if j, ok := net.cf.(JSONAble); ok {
		if err := saveJSON(j.Get(), dirPath + "/" + cf_ext + ".txt"); err != nil {
			return errors.Wrapf(err, "Failed to save Network CostFunction\n")
		}
	}

	return nil
}

func (n *Node) writeFile(dirPath string) error {
	path := dirPath + "/" + strconv.Itoa(n.id)
	if _, err := os.Stat(path); err != nil {
		// 0700 indicates universal read/write permissions
		if err = os.MkdirAll(path, 0700); err != nil {
			return errors.Wrapf(err, "Failed to create save directory for Node %v\n", n)
		}
	}

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

	pen := ""
	if n.pen != nil {
		pen = n.pen.TypeString()
	}

	p := proxy_Node{
		Size:       n.Size(),
		Name:       n.Name(),
		TypeString: typStr,
		OptString:  optStr,
		HPTypes:    hpTypes,
		PenString:  pen,
		InputsID:   nodesToIDs(n.inputs.nodes),
		Delay:      n.Delay(),
	}

	if err := saveJSON(p, path + ".node"); err != nil {
		return errors.Wrapf(err, "Failed to save Node\n")
	}

	if n.typ != nil {
		if st, ok := n.typ.(Storable); ok {
			if err := st.Save(path + "/" + typ_ext); err != nil {
				return errors.Wrapf(err, "Failed to save Operator\n")
			}
		} else if j, ok := n.typ.(JSONAble); ok {
			if err := saveJSON(j.Get(), path + "/" + typ_ext + ".txt"); err != nil {
				return errors.Wrapf(err, "Failed to save Operator\n")
			}
		}
	}

	if n.opt != nil {
		if st, ok := n.opt.(Storable); ok {
			if err := st.Save(path + "/" + opt_ext); err != nil {
				return errors.Wrapf(err, "Failed to save Optimizer\n")
			}
		} else if j, ok := n.opt.(JSONAble); ok {
			if err := saveJSON(j.Get(), path + "/" + opt_ext + ".txt"); err != nil {
				return errors.Wrapf(err, "Failed to save Optimizer\n")
			}
		}
	}

	for name, hp := range n.hyperParams {
		if st, ok := hp.(Storable); ok {
			if err := st.Save(path + "/" + hp_pref + name); err != nil {
				return errors.Wrapf(err, "Failed to save HyperParameter (name: %s, type: %s)\n", name, hp.TypeString())
			}
		} else if j, ok := hp.(JSONAble); ok {
			if err := saveJSON(j.Get(), path + "/" + hp_pref + name + ".txt"); err != nil {
				return errors.Wrapf(err, "Failed to save HyperParameter (name: %s, type: %s)\n", name, hp.TypeString())
			}
		}
	}

	if n.pen != nil {
		if st, ok := n.pen.(Storable); ok {
			if err := st.Save(path + "/" + pen_ext); err != nil {
				return errors.Wrapf(err, "Failed to save Penalty\n")
			}
		} else if j, ok := n.pen.(JSONAble); ok {
			if err := saveJSON(j.Get(), path + "/" + pen_ext + ".txt"); err != nil {
				return errors.Wrapf(err, "Failed to save Penalty\n")
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

// Load generates the Network from the provided directory, written to by Save
func Load(dirPath string) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(dirPath); err != nil {
		return nil, errors.Errorf("Containing directory does not exist")
	}

	// Load the main network
	var pNet proxy_Network
	if err := loadJSON(&pNet, dirPath + "/" + main_file); err != nil {
		return nil, errors.Wrapf(err, "Failed to load Network overview\n")
	}

	net := new(Network)
	pNodes := make([]proxy_Node, pNet.NumNodes)
	net.iter = pNet.Iter

	// Load the Nodes, make placeholders
	for id := 0; id < pNet.NumNodes; id++ {
		path := dirPath + "/" + strconv.Itoa(id)

		if err := loadJSON(&pNodes[id], path + ".node"); err != nil {
			return nil, errors.Wrapf(err, "Failed to read save for Node (id %d)\n", id)
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
		var pen Penalty
		if len(inputs) > 0 {
			if ops[pNodes[id].TypeString] == nil {
				return nil, errors.Errorf("Operator type %q has not been registered", pNodes[id].TypeString)
			}

			typ = ops[pNodes[id].TypeString]()
			if st, ok := typ.(Storable); ok {
				if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + typ_ext); err != nil {
					return nil, errors.Wrapf(err, "Failed to Load Operator for Node %q (id %d)\n", pNodes[id].Name, id)
				}
			} else if j, ok := typ.(JSONAble); ok {
				if err := loadJSON(j.Blank(), dirPath + "/" + strconv.Itoa(id) + "/" + typ_ext + ".txt"); err != nil {
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
						return nil, errors.Wrapf(err, "Failed to load Optimizer for Node %q (id %d)\n", pNodes[id].Name, id)
					}
				} else if j, ok := opt.(JSONAble); ok {
					if err := loadJSON(j.Blank(), dirPath + "/" + strconv.Itoa(id) + "/" + opt_ext + ".txt"); err != nil {
						return nil, errors.Wrapf(err, "Failed to load Optimizer for Node %q (id %d)\n", pNodes[id].Name, id)
					}
				}

				if pNodes[id].PenString != "" {
					pen = pens[pNodes[id].PenString]()
					if st, ok := pen.(Storable); ok {
						if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + pen_ext); err != nil {
							return nil, errors.Wrapf(err, "Failed to Load Penalty for Node %q (id %d)\n", pNodes[id].Name, id)
						}
					} else if j, ok := pen.(JSONAble); ok {
						if err := loadJSON(j.Blank(), dirPath + "/" + strconv.Itoa(id) + "/" + pen_ext + ".txt"); err != nil {
							return nil, errors.Wrapf(err, "Failed to load Penalty for Node %q (id %d)\n", pNodes[id].Name, id)
						}
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
		if pen != nil {
			n.SetPenalty(pen)
		}

		for name, ts := range pNodes[id].HPTypes {
			if hps[ts] == nil {
				return nil, errors.Errorf("HyperParameter type %q has not been registered", ts)
			}

			hp := hps[ts]()

			if st, ok := hp.(Storable); ok {
				if err := st.Load(dirPath + "/" + strconv.Itoa(id) + "/" + hp_pref + name); err != nil {
					return nil, errors.Wrapf(err, "Failed to Load HyperParameter %q for Node %q (id %d)\n", ts, n.name, id)
				}
			} else if j, ok := hp.(JSONAble); ok {
				if err := loadJSON(j.Blank(), dirPath + "/" + strconv.Itoa(id) + "/" + hp_pref + name + ".txt"); err != nil {
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
	if st, ok := cf.(Storable); ok {
		if err := st.Load(dirPath + "/" + cf_ext); err != nil {
			return nil, errors.Wrapf(err, "Failed to Load Network CostFunction\n")
		}
	} else if j, ok := cf.(JSONAble); ok {
		if err := loadJSON(j.Blank(), dirPath + "/" + cf_ext + ".txt"); err != nil {
			return nil, errors.Wrapf(err, "Failed to load Network CostFunction\n")
		}
	}

	if err := net.finalize(true, cf, idsToNodes(net.nodesByID, pNet.OutputsID)...); err != nil {
		return nil, errors.Wrapf(err, "Failed to finalize Network\n")
	}

	return net, nil
}

// Graph generates a graph of the Network, through the DOT Language and graphviz's
// dot command. N.B: if dot is not installed, this method will fail. Graph creates a
// pdf file with the name and location given by path.
//
// Specifics: The graph printed is a digraph, with dotted lines for connections with
// delay, labeled with the amount of delay if it is more than 1.
//
// Graphviz is available at https://graphviz.gitlab.io/
func (net *Network) Graph(path string) error {
	// dot -Tpdf /dev/stdin -o graph.pdf
	cmd := exec.Command("dot", "-Tpdf", "/dev/stdin", "-o", path+".pdf")

	// error only occurs if stdin is set or if the process has started
	writer, err := cmd.StdinPipe()
	if err != nil {
		return errors.Wrapf(err, "Failed to set StdinPipe for command\n")
	}

	if err := cmd.Start(); err != nil {
		return errors.Wrapf(err, "Failed to start dot command\n")
	}

	// write file to pipe
	func() {
		print := func(format string, a ...interface{}) error {
			_, err := fmt.Fprintf(writer, format+"\n", a...)
			return err
		}

		print("digraph {")

		for id, n := range net.nodesByID {
			if err := print("%d [label=%q]", id, n.String()); err != nil {
				return
			}

			if num(n.outputs) != 0 {
				for _, o := range n.outputs.nodes {
					if n.Delay() != 0 {
						if err := print("%d -> %d [style=\"dashed\", label=\"%d\"]", id, o.id, n.Delay()); err != nil {
							return
						}
					} else {
						if err := print("%d -> %d", id, o.id); err != nil {
							return
						}
					}
				}
			}
		}

		if err := print("}"); err != nil {
			return
		}

		writer.Close()
	}()

	if err := cmd.Wait(); err != nil {
		return errors.Wrapf(err, "dot command failed\n")
	}

	return nil
}
