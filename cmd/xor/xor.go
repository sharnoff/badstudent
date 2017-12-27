package main

import (
	learn "github.com/sharnoff/smartlearning/smartlearning"
	"github.com/pkg/errors"
	"fmt"
)

func data() []*learn.Datum {
	d := make([]*learn.Datum, 4)
	d[0] = new(learn.Datum)
	d[0].Inputs = []float64{0, 0}
	d[0].Outputs = []float64{0}
	d[1] = new(learn.Datum)
	d[1].Inputs = []float64{0, 1}
	d[1].Outputs = []float64{1}
	d[2] = new(learn.Datum)
	d[2].Inputs = []float64{1, 0}
	d[2].Outputs = []float64{1}
	d[3] = new(learn.Datum)
	d[3].Inputs = []float64{1, 1}
	d[3].Outputs = []float64{0}
	return d
}

func main() {
	fmt.Print("Initializing network...")
	net, err := learn.New(2, 1)
	if err != nil {
		fmt.Printf("%s\n", errors.Wrap(err, "couldn't create new network\n"))
		return
	}

	{
		l, err := net.Add("inputs", learn.Blank(), []int{2})
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - inputs\n", err.Error())
		}

		hl, err := net.Add("hl", learn.Neurons(), []int{1}, l)
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - hl\n", err.Error())
		}

		outs, err := net.Add("outputs", learn.Neurons(), []int{1}, l, hl)
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - outputs", err.Error())
		}

		err = net.SetOutputs(outs)
		if err != nil {
			fmt.Printf("%s\n: couldn't set outputs to network", err.Error())
		}
	}

	/*{
		l, err := net.Add("inputs", learn.Blank(), []int{2})
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network - inputs\n"))
			return
		}
		l, err = net.Add("hl", learn.Neurons(), []int{3}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network - hl\n"))
			return
		}
		l, err = net.Add("outputs", learn.Neurons(), []int{1}, l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't fill in new network - outputs\n"))
			return
		}
		err = net.SetOutputs(l)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrap(err, "couldn't set outputs to network\n"))
			return
		}
	}*/

	fmt.Println(" Done!")

	fmt.Print("Fetching data...")
	trainData := data()
	fmt.Println(" Done!")

	learningRate := 0.01
	maxEons := 100

	fmt.Printf("Starting training for %d eons\n", maxEons)
	var trainE float64
	for eon := 1; eon <= maxEons; eon++ {
		trainE, _, err = net.Train(trainData, learningRate)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrapf(err, "error in training during eon %d\n", eon))
			return
		}
		fmt.Printf("%v\n", trainE)
	}
	
	fmt.Println("Done training... performing final tests")


	for i, d := range trainData {
		outs, err := net.GetOutputs(d.Inputs)
		if err != nil {
			fmt.Printf("%s\n\t- in mnist.main() at test %d", err.Error(), i)
		}
		fmt.Printf("%v → %v\n", d.Outputs, outs)
	}
}