package hyperparams

type constant float64

func Constant(value float64) *constant {
	c := constant(value)
	return &c
}

func (c constant) TypeString() string {
	return "constant"
}

func (c *constant) Value(iter int) float64 {
	return *(*float64)(c)
}

func (c *constant) Get() interface{} {
	return *c
}

func (c *constant) Blank() interface{} {
	return c
}
