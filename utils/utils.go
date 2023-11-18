package utils

import (
	"ann/matrix"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func ReadCSV(file_name string) [](matrix.Matrix) {
	file, err := os.Open(file_name)
	checkError(err, "Error opening file:")
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	checkError(err, "Error reading CSV:")

	var content [](matrix.Matrix)

	for i, row := range records {
		content = append(content, matrix.NewMatrix(len(row), 1))
		// X[0].Values = [][]float64{{0}, {0}} //
		content[i].Values = stringToFloat(row)
	}
	return content
}

func stringToFloat(str []string) [][]float64 {
	var flts [][]float64
	for i := 0; i < len(str); i++ {
		flt, err := strconv.ParseFloat(str[i], 64)
		checkError(err, "error converting string to float")
		flts = append(flts, []float64{flt})
	}
	return flts
}

func checkError(err error, message string) {
	if err != nil {
		fmt.Println(message, err)
		return
	}
}
