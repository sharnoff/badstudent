// converts the csv files to arrays of images
package main

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
	"os"
)

const (
	imgSize    int = 784 // 28x28
	numOptions int = 10  // 0->9
)

type image struct {
	data  []int // this is just the inputs to the network
	label []int // this should all 0s, except a 1 at the index of the correct value
}

func getImgsFromFile(fileName string) ([]*image, error) {
	funcName := "mnist.getImgsFromFile()"

	numLines, err := numLines(fileName)
	if err != nil {
		return nil, fmt.Errorf("%s\n\t- in %s", err.Error(), funcName)
	}

	file, err := os.Open(fileName)
	if err != nil {
		return nil, fmt.Errorf("%s\n\t- in %s", err.Error(), funcName)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	imgs := make([]*image, numLines)
	for i := 0; scanner.Scan(); i++ {
		imgs[i], err = getImage(scanner.Text())
		if err != nil {
			return nil, fmt.Errorf("%s\n\t- in %s at image %d", err.Error(), funcName, i)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%s\n\t- in %s", err.Error(), funcName)
	}

	return imgs, nil
}

func numLines(fileName string) (int, error) {
	funcName := "mnist.numLines()"

	file, err := os.Open(fileName)
	if err != nil {
		return 0, fmt.Errorf("%s\n\t- in %s", err.Error(), funcName)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	numLines := 0
	for scanner.Scan() {
		numLines++
	}

	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("%s\n\t- in %s", err.Error(), funcName)
	}

	return numLines, nil
}

func getImage(s string) (*image, error) {
	funcName := "mnist.getImage()"

	img := new(image)
	img.data = make([]int, imgSize)
	img.label = make([]int, numOptions)

	strs := strings.Split(s, ",")

	lbl, err := strconv.ParseInt(strs[0], 10, 0)
	if err != nil {
		return nil, fmt.Errorf("%s\n\t- in %s - strconv.ParseInt(%s)", err.Error(), funcName, strs[0])
	}
	img.label[lbl] = 1

	for i := range img.data {
		v, err := strconv.ParseInt(strs[i+1], 10, 0)
		if err != nil {
			return nil, fmt.Errorf("%s\n\t- in %s - strconv.ParseInt(%s)", err.Error(), funcName, strs[i+1])
		}
		img.data[i] = int(v)
	}

	return img, nil
}
