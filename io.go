package badstudent

import (
	"encoding/json"
	"fmt"
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

type proxyNetwork struct {
	OutputsID []int
	NumNodes  int
	Iter      int
	CFString  string
	HPStrings map[string]string
	PenString string
}

type proxyNode struct {
	Dims      []int
	Name      string
	OpString  string
	OptString string
	HPStrings map[string]string
	PenString string
	InputsID  []int
	Delay     int
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
	node_ext  string = ".node"
	cf_ext    string = "cf"
	op_ext    string = "op"
	opt_ext   string = "opt"
	hp_pref   string = "hp_"
	pen_ext   string = "pen"
)

// FileError stores errors from attempting to access files, either to write to them or to read
// them.
type FileError struct {
	Path string
	Err  string
}

func (err FileError) Error() string {
	return err.Err + " at " + err.Path
}

func saveJSON(v interface{}, path string, addTxt bool) error {
	if addTxt {
		path += ".txt"
	}

	f, err := os.Create(path)
	if err != nil {
		return FileError{path, "Failed to create file"}
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	if err = enc.Encode(v); err != nil {
		return FileError{path, "Failed to encode JSON"}
	}

	return nil
}

func loadJSON(v interface{}, path string, addTxt bool) error {
	if addTxt {
		path += ".txt"
	}

	f, err := os.Open(path)
	if err != nil {
		return FileError{path, "Failed to open file"}
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	if err = dec.Decode(v); err != nil {
		return FileError{path, "Failed to decode JSON"}
	}

	return nil
}

func saveElement(v interface{}, path string) error {
	if st, ok := v.(Storable); ok {
		if err := st.Save(path); err != nil {
			return err
		}
	} else if j, ok := v.(JSONAble); ok {
		if err := saveJSON(j.Get(), path, true); err != nil {
			return err
		}
	}

	return nil
}

func loadElement(v interface{}, path string) error {
	if st, ok := v.(Storable); ok {
		if err := st.Load(path); err != nil {
			return err
		}
	} else if j, ok := v.(JSONAble); ok {
		if err := loadJSON(j.Blank(), path, true); err != nil {
			return err
		}
	}

	return nil
}

// FieldIOError is a wrapper for other errors ocurring for saving or loading parts of a Network due
// to direct interfacing with files.
type FieldIOError struct {
	// ContainingStruct indicates which (either Node or Network) the i/o error occured with. Its
	// value is then either "Network" or "Node"
	ContainingStruct string

	// Field indicates the field of ContainingStruct that was being saved/loaded. Field will be
	// equal to an empty string if the thing being saved/loaded is the ContainingStruct itself
	Field string

	// Op indicates whether the value was being saved or loaded. The value of Op is then either
	// "save" or "load"
	Op string

	Err error
}

func (err FieldIOError) Error() string {
	if err.Field == "" {
		return "Failed to " + err.Op + " " + err.ContainingStruct + ": " + err.Err.Error()
	}

	return "Failed to " + err.Op + " " + err.ContainingStruct + " " + err.Field + ": " + err.Err.Error()
}

func (net *Network) writeFile(dirPath string) error {
	p := proxyNetwork{
		OutputsID: nodesToIDs(net.outputs.nodes),
		NumNodes:  len(net.nodesByID),
		Iter:      net.iter,
		CFString:  net.cf.TypeString(),
	}

	if net.pen != nil {
		p.PenString = net.pen.TypeString()
	}

	p.HPStrings = make(map[string]string)
	for name, hp := range net.hyperParams {
		p.HPStrings[name] = hp.TypeString()
	}

	if err := saveJSON(p, dirPath+"/"+main_file, false); err != nil {
		return FieldIOError{"Network", "", "save", err}
	}

	if err := saveElement(net.cf, dirPath+"/"+cf_ext); err != nil {
		return FieldIOError{"Network", "CostFunction", "save", err}
	}

	if net.pen != nil {
		if err := saveElement(net.pen, dirPath+"/"+pen_ext); err != nil {
			return FieldIOError{"Network", "Penalty", "save", err}
		}
	}

	for name, hp := range net.hyperParams {
		if err := saveElement(hp, dirPath+"/"+hp_pref+name); err != nil {
			return FieldIOError{"Network", "HyperParameter (" + name + ")", "save", err}
		}
	}

	return nil
}

func (n *Node) writeFile(dirPath string) error {
	path := dirPath + "/" + strconv.Itoa(n.id)
	if _, err := os.Stat(path); err != nil {
		// 0700 indicates universal read/write permissions
		if err = os.MkdirAll(path, 0700); err != nil {
			return FileError{path, "Could not make directory to save Node"}
		}
	}

	/*
		type proxyNode struct {
			Dims      []int
			Name      string
			OpString  string
			OptString string
			HPStrings map[string]string
			PenString string
			InputsID  []int
			Delay     int
		}
	*/

	var p proxyNode
	{
		p = proxyNode{
			Dims:  n.values.Dims,
			Name:  n.name,
			Delay: n.Delay(),
		}

		if !n.IsInput() {
			p.InputsID = nodesToIDs(n.inputs.nodes)

			p.OpString = n.op.TypeString()
			if n.opt != nil {
				p.OptString = n.opt.TypeString()
			}
		}

		p.HPStrings = make(map[string]string)
		for name, hp := range n.hyperParams {
			p.HPStrings[name] = hp.TypeString()
		}

		if n.pen != nil {
			p.PenString = n.pen.TypeString()
		}
	}

	if err := saveJSON(p, path+node_ext, false); err != nil {
		return FieldIOError{"Node " + n.String(), "", "save", err}
	}

	if n.op != nil {
		if err := saveElement(n.op, path+"/"+op_ext); err != nil {
			return FieldIOError{"Node " + n.String(), "Operator", "save", err}
		}
	}

	if n.opt != nil {
		if err := saveElement(n.opt, path+"/"+opt_ext); err != nil {
			return FieldIOError{"Node " + n.String(), "Optimizer", "save", err}
		}
	}

	if n.pen != nil {
		if err := saveElement(n.pen, path+"/"+pen_ext); err != nil {
			return FieldIOError{"Node " + n.String(), "Penalty", "save", err}
		}
	}

	for name, hp := range n.hyperParams {
		if err := saveElement(hp, path+"/"+hp_pref+name); err != nil {
			return FieldIOError{"Node " + n.String(), "HyperParameter (" + name + ")", "save", err}
		}
	}

	return nil
}

func (net *Network) Save(path string, overwrite bool) (bool, error) {
	// check if the folder already exists
	if _, err := os.Stat(path); err == nil {
		if !overwrite {
			return false, nil
		}

		if err := os.RemoveAll(path); err != nil {
			return false, FileError{path, ""} // failed to remove
		}
	}

	if err := os.MkdirAll(path, 0700); err != nil {
		return false, FileError{path, "Failed to create save directory"} // failed to create save dir
	}

	var sucessful bool
	defer func() {
		if !sucessful {
			os.RemoveAll(path)
		}
	}()

	if err := net.writeFile(path); err != nil {
		return false, err
	}

	for _, n := range net.nodesByID {
		if err := n.writeFile(path); err != nil {
			return false, err
		}
	}

	sucessful = true
	return true, nil
}

// NotRegisteredError documents errors from types that have not been registered.
type NotRegisteredError struct {
	Typ string
	Str string
}

func (err NotRegisteredError) Error() string {
	return err.Typ + " type \"" + err.Str + "\" has not been registered"
}

// ConstructionError documents errors occuring from the re-construction of the Network from a saved
// version.
type ConstructionError struct {
	Func     string
	NodeName string
	Err      error
}

func (err ConstructionError) Error() string {
	if err.NodeName == "" {
		return "Construction error from " + err.Func + ": " + err.Err.Error()
	}

	return "Construction error from " + err.Func + " with Node " + err.NodeName + ": " + err.Err.Error()
}

// Load generates a Network from a previously saved version.
func Load(path string) (*Network, error) {
	// check if the folder exists
	if _, err := os.Stat(path); err != nil {
		return nil, FileError{path, "Previously saved directory does not exist"} // containing directory does not exist
	}

	var pNet proxyNetwork
	if err := loadJSON(&pNet, path+"/"+main_file, false); err != nil {
		return nil, FieldIOError{"Network", "", "load", err} // failed to load network overview
	}

	net := new(Network)
	pNodes := make([]proxyNode, pNet.NumNodes)
	net.iter = pNet.Iter

	// Load the Nodes
	for id := 0; id < pNet.NumNodes; id++ {
		path := path + "/" + strconv.Itoa(id) + node_ext

		if err := loadJSON(&pNodes[id], path, false); err != nil {
			return nil, FieldIOError{"Node (id: " + strconv.Itoa(id) + ")", "", "load", err} // failed to read save for Node (id %d)
		}
	}

	// add all input Nodes, make placeholders for non-inputs
	for id, pn := range pNodes {
		var n *Node

		if len(pn.InputsID) == 0 {
			n = net.AddInput(pn.Dims)
			if net.Error() != nil {
				var name string
				if pn.Name != "" {
					name = "\"" + pn.Name + "\""
				} else {
					name = fmt.Sprintf("<id: %d>", id)
				}

				return nil, ConstructionError{"AddInput", name, net.Error()}
			}
		} else {
			n = net.Placeholder(pn.Dims)
			if net.Error() != nil {
				var name string
				if pn.Name != "" {
					name = "\"" + pn.Name + "\""
				} else {
					name = fmt.Sprintf("<id: %d, Operator: %s>", id, pn.OpString)
				}

				return nil, ConstructionError{"Placeholder", name, net.Error()}
			}
		}

		if pn.Name != "" {
			n.SetName(pn.Name)
		}
	}

	// replace placeholders
	for id, n := range net.nodesByID {
		if n.IsInput() {
			continue
		}

		path := path + "/" + strconv.Itoa(id)
		pn := pNodes[id]

		var name string
		if pn.Name != "" {
			name = "\"" + pn.Name + "\""
		} else {
			name = fmt.Sprintf("<id: %d, Operator: %s>", id, pn.OpString)
		}

		// replace Node
		{
			var op Operator
			var opGen func() Operator
			if opGen = ops[pn.OpString]; opGen == nil {
				return nil, NotRegisteredError{"Operator", pn.OpString}
			} else if op = opGen(); op == nil {
				return nil, ErrRegisterNilReturn
			}

			if err := loadElement(op, path+"/"+op_ext); err != nil {
				return nil, FieldIOError{"Node (" + name + ")", "Operator", "load", err}
			}

			inputs := idsToNodes(net.nodesByID, pn.InputsID)
			n.Replace(op, inputs...)

			if net.Error() != nil {
				return nil, ConstructionError{"Replace", name, net.Error()}
			}
		}

		// add extras
		{
			n.SetDelay(pn.Delay)

			if pn.OptString != "" {
				var opt Optimizer
				var optGen func() Optimizer
				if optGen = opts[pn.OptString]; optGen == nil {
					return nil, NotRegisteredError{"Optimizer", pn.OptString}
				} else if opt = optGen(); opt == nil {
					return nil, ErrRegisterNilReturn
				}

				if err := loadElement(opt, path+"/"+opt_ext); err != nil {
					return nil, FieldIOError{"Node (" + name + ")", "Optimizer", "load", err}
				}

				n.Opt(opt)

				// Because Opt will only set net.Error() if opt == nil, and we've already shown
				// that it's not, we dont' actually need to check whether or not net.Error() is
				// nil.
			}

			if pn.PenString != "" {
				var pen Penalty
				var penGen func() Penalty
				if penGen = pens[pn.PenString]; penGen == nil {
					return nil, NotRegisteredError{"Penalty", pn.PenString}
				} else if pen = penGen(); pen == nil {
					return nil, ErrRegisterNilReturn
				}

				if err := loadElement(pen, path+"/"+pen_ext); err != nil {
					return nil, FieldIOError{"Node (" + name + ")", "Penalty", "load", err}
				}

				n.SetPenalty(pen)

				// Because SetPenalty will only set net.Error() if pen == nil, and we've already
				// shown that it's not, we don't actually need to check whether or not net.Error()
				// is nil.
			}

			for hpName, typ := range pn.HPStrings {
				var hp HyperParameter
				var hpGen func() HyperParameter
				if hpGen = hps[typ]; hpGen == nil {
					return nil, NotRegisteredError{"HyperParameter (" + hpName + ")", typ}
				} else if hp = hpGen(); hp == nil {
					return nil, ErrRegisterNilReturn
				}

				if err := loadElement(hp, path+"/"+hp_pref+hpName); err != nil {
					return nil, FieldIOError{"Node (" + name + ")", "HyperParameter (" + hpName + ")", "load", err}
				}

				n.AddHP(name, hp)

				// AddHP can set net.Error() to one of two errors:
				//	NilArgError, if hp == nil, or
				//	ErrHPNameTaken, if name has already been used.
				// Both of these are not possible at this stage because we've checked for them
				// already. The name of the HyperParameter cannot be re-used because the map keys
				// for pn.HPStrings are the names.
			}
		}
	}

	// set Network penalties and HyperParameters, if it has them:
	{
		if pNet.PenString != "" {
			var pen Penalty
			var penGen func() Penalty
			if penGen = pens[pNet.PenString]; penGen == nil {
				return nil, NotRegisteredError{"Penalty", pNet.PenString}
			} else if pen = penGen(); pen == nil {
				return nil, ErrRegisterNilReturn
			}

			if err := loadElement(pen, path+"/"+pen_ext); err != nil {
				return nil, FieldIOError{"Network", "Penalty", "load", err}
			}

			net.SetPenalty(pen)

			// Because SetPenalty will only set net.Error() if pen == nil, and we've already shown
			// that it's not, we don't actually need to check whether or not net.Error() is nil.
		}

		for name, typ := range pNet.HPStrings {
			var hp HyperParameter
			var hpGen func() HyperParameter
			if hpGen = hps[typ]; hpGen == nil {
				return nil, NotRegisteredError{"HyperParameter", typ}
			} else if hp = hpGen(); hp == nil {
				return nil, ErrRegisterNilReturn
			}

			if err := loadElement(hp, path+"/"+hp_pref+name); err != nil {
				return nil, FieldIOError{"Network", "HyperParameter (" + name + ")", "load", err}
			}

			net.AddHP(name, hp)

			// AddHP can set net.Error() to one of two errors:
			//	NilArgError, if hp == nil, or
			//	ErrHPNameTaken, if name has already been used.
			// Both of these are not possible at this stage because we've checked for them
			// already. The name of the HyperParameter cannot be re-used because the map keys
			// for pn.HPStrings are the names.
		}
	}

	// finish up making the Network
	{
		var cf CostFunction
		var cfGen func() CostFunction
		if cfGen = cfs[pNet.CFString]; cfGen == nil {
			return nil, NotRegisteredError{"CostFunction", pNet.CFString}
		} else if cf = cfGen(); cf == nil {
			return nil, ErrRegisterNilReturn
		}

		if err := loadElement(cf, path+"/"+cf_ext); err != nil {
			return nil, FieldIOError{"Network", "CostFunction", "load", err}
		}

		if err := net.finalize(true, cf, idsToNodes(net.nodesByID, pNet.OutputsID)...); err != nil {
			return nil, ConstructionError{"Finalize", "", err}
		}
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

	writer, err := cmd.StdinPipe()
	if err != nil {
		return ErrFailedCommand
	}

	if err := cmd.Start(); err != nil {
		return ErrFailedCommand // failed to start dot command
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
		return ErrFailedCommand
	}

	return nil
}
