package hyperparams

type step struct {
	Iter int
	Val  float64
}

type stepper []step

func Step(base float64) *stepper {
	s := make([]step, 1)

	s[0] = step{0, base}

	st := stepper(s)
	return &st
}

// Add adds a step to the HyperParameter.
func (s *stepper) Add(iter int, value float64) *stepper {
	*s = append(*s, step{iter, value})
	return s
}

func (s *stepper) TypeString() string {
	return "step"
}

func (s *stepper) Value(iter int) float64 {
	sl := []step(*s)
	for i := 1; i < len(sl); i++ {
		if sl[i].Iter > iter {
			return sl[i-1].Val
		}
	}

	return sl[len(sl)-1].Val
}

func (s *stepper) Get() interface{} {
	return *s
}

func (s *stepper) Blank() interface{} {
	return s
}
