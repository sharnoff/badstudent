package badstudent

import (
	"github.com/pkg/errors"
	"sort"
)

// Returns the number of values in the group
func (ng *nodeGroup) size() int {
	if len(ng.sumVals) == 0 {
		return 0
	}

	return ng.sumVals[len(ng.sumVals)-1]
}

// Returns the number of nodes in the group
func num(ng *nodeGroup) int {
	return len(ng.nodes)
}

// This method is self-explanatory
// Returns whether or not the nodes were added
//
// Addition is much more expensive if the nodeGroup is already
// continuous.
func (ng *nodeGroup) add(nodes ...*Node) bool {
	if ng.values != nil {
		for _, n := range nodes {
			if n.group != nil {
				return false
			}
		}
	}

	ng.nodes = append(ng.nodes, nodes...)

	// could be improved but is good enough as is:
	for i := range nodes {
		ng.sumVals = append(ng.sumVals, ng.size()+nodes[i].Size())

		if ng.values != nil {
			nodes[i].group = ng
		}
	}

	if ng.values != nil {
		ng.values = make([]float64, ng.sumVals[len(ng.sumVals)-1])
		for i, n := range ng.nodes {
			n.values = ng.values[ng.sumVals[i]-n.Size() : ng.sumVals[i]]
		}
	}

	return true
}

// Returns whether or not the nodeGroup was able to be made continuous.
// Cannot become continuous if a Node is already in another continuous
// nodeGroup.
//
// This function should not be run in parallel with overlapping members.
func (ng *nodeGroup) makeContinuous() bool {
	// check that none of the nodes are already taken
	for _, n := range ng.nodes {
		if n.group != nil {
			return false
		}
	}

	ng.values = make([]float64, ng.sumVals[len(ng.sumVals)-1])
	for i, n := range ng.nodes {
		n.values = ng.values[ng.sumVals[i]-n.Size() : ng.sumVals[i]]
		n.group = ng
	}

	return true
}

func (ng *nodeGroup) isContinuous() bool {
	return (ng.values != nil)
}

// Duplicates the internal slices to reduce unused capacity
func (ng *nodeGroup) trim() {
	nodes := make([]*Node, len(ng.nodes))
	copy(nodes, ng.nodes)
	ng.nodes = nodes

	sumVals := make([]int, len(ng.sumVals))
	copy(sumVals, ng.sumVals)
	ng.sumVals = sumVals

	return
}

// Sets the values of the nodeGroup and each Node's status to 'inputsChanged'
func (ng *nodeGroup) setValues(values []float64) error {
	if ng.sumVals[len(ng.sumVals)-1] != len(values) {
		return errors.Errorf("Number of given values and group values don't match (%d != %d)", len(values), ng.sumVals[len(ng.sumVals)-1])
	}

	if ng.values != nil {
		copy(ng.values, values)
	} else {
		for i, n := range ng.nodes {
			n.values = values[ng.sumVals[i]-n.Size() : ng.sumVals[i]]
		}
	}

	return nil
}

// Returns the values of the nodeGroup
// Will need to copy if the nodeGroup is not continuous
func (ng *nodeGroup) getValues(dupe bool) []float64 {
	if ng.isContinuous() {
		if dupe {
			values := make([]float64, ng.size())
			copy(values, ng.values)
			return values
		} else {
			return ng.values
		}
	} else { // not continuous
		values := make([]float64, ng.size())
		for i, n := range ng.nodes {
			copy(values[ng.sumVals[i]-n.Size():], n.values)
		}
		return values
	}
}

// assumes len(ds) == ng.size()
//
// not multithreaded -- could be a possible improvement
func (ng *nodeGroup) addDeltas(ds []float64) {
	for i, n := range ng.nodes {
		if len(n.deltas) == 0 {
			continue
		}

		// position in ds
		pos := ng.sumVals[i] - n.Size()

		d := n.deltas
		if n.HasDelay() {
			d = n.tempDelayDeltas
		}

		for j := range d {
			d[j] += ds[pos+j]
		}
	}
}

// If not continuous, binary searches for the node with the specified index
// Allows out-of-bounds panics instead of returning a nil error
func (ng *nodeGroup) value(index int) float64 {
	// Note: an easy optimization would be to add special cases for one or two nodes
	if ng.isContinuous() {
		return ng.values[index]
	}

	greaterThan := func(i int) bool {
		return index < ng.sumVals[i]
	}

	i := sort.Search(len(ng.nodes), greaterThan)

	if i > 0 {
		index -= ng.sumVals[i-1]
	}

	return ng.nodes[i].Value(index)
}

// Returns a channel that iterates over the values of the group
func (ng *nodeGroup) valueIterator() chan float64 {
	ch := make(chan float64)
	if ng.isContinuous() {
		go func() {
			for _, v := range ng.values {
				ch <- v
			}
		}()
	} else {
		go func() {
			for _, n := range ng.nodes {
				for v := range n.valueIterator() {
					ch <- v
				}
			}
			close(ch)
		}()
	}

	return ch
}
