package main

import (
	"github.com/sharnoff/smartlearning/smartlearn"
	"github.com/sharnoff/smartlearning/smartlearn/segmenttypes"
	"github.com/pkg/errors"
	"fmt"
)

func data() []*smartlearn.Datum {
	d := make([]*smartlearn.Datum, 4)
	d[0] = new(smartlearn.Datum)
	d[0].Inputs = []float64{0, 0}
	d[0].Outputs = []float64{0}
	d[1] = new(smartlearn.Datum)
	d[1].Inputs = []float64{0, 1}
	d[1].Outputs = []float64{1}
	d[2] = new(smartlearn.Datum)
	d[2].Inputs = []float64{1, 0}
	d[2].Outputs = []float64{1}
	d[3] = new(smartlearn.Datum)
	d[3].Inputs = []float64{1, 1}
	d[3].Outputs = []float64{0}
	return d
}

func main() {
	net := new(smartlearn.Network)

	fmt.Println("Initializing network...")
	{
		l, err := net.Add("inputs", segmenttypes.Blank(), []int{2})
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - inputs\n", err.Error())
			return
		}

		hl, err := net.Add("hl", segmenttypes.Neurons(), []int{1}, l)
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - hl\n", err.Error())
			return
		}

		outs, err := net.Add("outputs", segmenttypes.Neurons(), []int{1}, l, hl)
		if err != nil {
			fmt.Printf("%s\n: couldn't fill in new network - outputs\n", err.Error())
			return
		}

		err = net.SetOutputs(outs)
		if err != nil {
			fmt.Printf("%s\n: couldn't set outputs to network\n", err.Error())
			return
		}
	}
	fmt.Println(" Done!")

	fmt.Print("Fetching data...")
	trainData := data()
	fmt.Println(" Done!")

	learningRate := 0.01
	maxEons := 100

	fmt.Printf("Starting training for %d eons\n", maxEons)
	var err error
	var trainE, trainP float64
	for eon := 1; eon <= maxEons; eon++ {
		trainE, trainP, err = net.Train(trainData, learningRate)
		if err != nil {
			fmt.Printf("%s\n", errors.Wrapf(err, "error in training during eon %d\n", eon))
			return
		}
		fmt.Printf("%v, %v\n", trainE, trainP)
	}
	
	fmt.Println("Done training... performing final tests")


	for i, d := range trainData {
		outs, err := net.GetOutputs(d.Inputs, true)
		if err != nil {
			fmt.Printf("%s\n\t- in mnist.main() at test %d", err.Error(), i)
		}
		fmt.Printf("%v → %v\n", d.Outputs, outs)
	}
}