// commands.go
// used exclusively in segments.go
// and mainly in (*Segment).run()

package smartlearn

type command int

// 'com' is the command to be executed
// 'res' is short for 'result', and is where any error from calling this will be returned to
// 'aux' is any auxilliary information, which depends on the command
type commandWrapper struct {
	com command
	res chan error
	aux []interface{}
}

const (
	// to prevent any command from being 0
	_ int = iota
	// commands used for setting up the network
	checkOutputs     command = iota
	allocate         command = iota
	finishAllocating command = iota
	setMethods       command = iota
	// commands used for running the network - might be a good idea to make a command to quit
	inputsChanged command = iota // inputsChanged is used as a sort of restart command
	evaluate      command = iota
	deltas        command = iota
	inputDeltas   command = iota
	adjust        command = iota
)

func (c command) String() string {
	switch c {
		case checkOutputs:
			return "checkOutputs"		
		case allocate:
			return "allocate"
		case finishAllocating:
			return "finishAllocating"
		case setMethods:
			return "setMethods"
		case inputsChanged:
			return "inputsChanged"
		case evaluate:
			return "evaluate"
		case deltas:
			return "deltas"
		case inputDeltas:
			return "inputDeltas"
		case adjust:
			return "adjust"
		default:
			return ""
	}
}

func (c command) isSetup() bool {
	return (c == checkOutputs || c == allocate || c == finishAllocating || c == setMethods || c == inputsChanged)
}

func (c command) isRunning() bool {
	return (c == inputsChanged || c == evaluate || c == deltas || c == inputDeltas || c == adjust)
}
