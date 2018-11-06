package initializers

type leCun struct {
	*varianceScaling
}

func LeCun() leCun {
	return leCun{VarianceScaling().In()}
}

type he struct {
	*varianceScaling
}

func He() he {
	return he{VarianceScaling().In().Factor(2)}
}

type xavier struct {
	*varianceScaling
}

func Xavier() xavier {
	return xavier{VarianceScaling().Avg()}
}

func Glorot() xavier {
	return Xavier()
}
