package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"sort"

	"github.com/ghodss/yaml"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	USAverageRainfall float64 = 38.1
)

type Input struct {
	AnnualExpenses  int     `json:"annual_expenses"`
	AnnualIncome    int     `json:"annual_income"`
	HomeEquity      float64 `json:"home_equity"`
	CandidateCities []City  `json:"candidate_cities"`
}

type City struct {
	Name       string     `json:"name"`
	Education  HighSchool `json:"education"`
	RealEstate RealEstate `json:"real_estate"`
	Taxes      Taxes      `json:"taxes"`
	Crime      Crime      `json:"crime"`
	Climate    Climate    `json:"climate"`
	Family     Family     `json:"family"`
	Livability Livability `json:"livability"`
}

type HighSchool struct {
	Name               string  `json:"name"`
	URL                string  `json:"url"`
	USNews             int64   `json:"usnews"`
	MathProficiency    float64 `json:"math"`
	ReadingProficiency float64 `json:"reading"`
	GraduationRate     float64 `json:"graduation"`
	CollegeReadiness   float64 `json:"college"`
}

type RealEstate struct {
	SampleURL     string `json:"sample"`
	MarketValue   int    `json:"market"`
	AssessedValue int    `json:"assessed"`
}

type Taxes struct {
	Property float64 `json:"property"`
	Sales    float64 `json:"sales"`
	Income   float64 `json:"income"`
}

type Crime struct {
	Violent  float64 `json:"violent"`
	Property float64 `json:"property"`
}

type Climate struct {
	SunnyDays  float64 `json:"sunny_days"`
	RainInches float64 `json:"rain_inches"`
	SnowInches float64 `json:"snow_inches"`
}

type Family struct {
	MilesToMargaret float64 `json:"miles_to_margaret"`
	MilesToNich     float64 `json:"miles_to_nich"`
	MilesToPeggy    float64 `json:"miles_to_peggy"`
	MilesToRyan     float64 `json:"miles_to_ryan"`
	MultipleSuites  float64 `json:"multiple_suites"`
}

type Livability struct {
	Politics       float64 `json:"politics"`
	Culture        float64 `json:"culture"`
	Running        float64 `json:"running"`
	WalkScore      float64 `json:"walk_score"`
	MilesToAirport float64 `json:"miles_to_airport"`
}

type ScoreSet struct {
	columnNames []string  // city names
	rowNames    []string  // axes
	rowWeights  []float64 // weight of each row
	scores      *mat.Dense
}

type ScoreFunc func(City) float64

type ScoreGoal int

const (
	BIGGER ScoreGoal = iota
	SMALLER
)

func NewScoreSet(axis string, cities []City, better ScoreGoal, f ScoreFunc) *ScoreSet {
	var result ScoreSet
	result.columnNames = make([]string, len(cities))
	result.rowNames = []string{axis}
	result.rowWeights = []float64{1.0}

	row := make([]float64, len(cities))
	for i, city := range cities {
		result.columnNames[i] = city.Name
		row[i] = f(city)
	}

	// z-score normalize the row
	mu, sigma := stat.MeanStdDev(row, nil)
	for i := range row {
		val := (row[i] - mu) / sigma
		if SMALLER == better {
			val = val * -1.0
		}
		row[i] = val
	}

	result.scores = mat.NewDense(1, len(cities), row)
	return &result
}

type scoredCity struct {
	Name  string
	Score float64
}

func (set *ScoreSet) Print() {
	log.Printf("%v", set.rowNames)
	fa := mat.Formatted(set.scores, mat.Prefix("    "), mat.Squeeze())
	log.Printf("data:\n    %v", fa)

	for i, axis := range set.rowNames {
		percent := fmt.Sprintf("%0.1f%%", 100.0*set.rowWeights[i])
		log.Printf("%-6s %s", percent, axis)
		row := mat.Row(nil, i, set.scores)

		scored := make([]scoredCity, len(set.columnNames))
		for j, cityName := range set.columnNames {
			scored[j].Name = cityName
			scored[j].Score = 100.0 * distuv.UnitNormal.CDF(row[j])
		}

		sort.Slice(scored, func(i, j int) bool {
			return scored[i].Score > scored[j].Score
		})
		for j := range scored {
			percent = fmt.Sprintf("%0.1f", scored[j].Score)
			log.Printf("    %-20s%4s%%", scored[j].Name, percent)
		}
	}

	log.Print("Final")
	{
		scored := make([]scoredCity, len(set.columnNames))
		for j, cityName := range set.columnNames {
			col := mat.Col(nil, j, set.scores)

			mu, _ := stat.MeanStdDev(col, set.rowWeights)

			scored[j].Name = cityName
			scored[j].Score = 100.0 * distuv.UnitNormal.CDF(mu)
		}
		sort.Slice(scored, func(i, j int) bool {
			return scored[i].Score > scored[j].Score
		})
		for j := range scored {
			percent := fmt.Sprintf("%0.1f", scored[j].Score)
			log.Printf("    %-20s%4s%%", scored[j].Name, percent)
		}
	}

}

func Merge(sets []*ScoreSet, weights []float64) *ScoreSet {
	if len(sets) < 2 {
		panic("Merge must have 2 or more ScoreSets")
	}
	if weights != nil {
		if len(sets) != len(weights) {
			panic("merge sets/weights length must match")
		}
		var total float64
		for _, weight := range weights {
			total += weight
		}
		if total != 1.0 {
			panic("merge weights must sum to 1.0")
		}
	}

	var merged ScoreSet
	merged.columnNames = sets[0].columnNames

	columns := sets[0].columnNames
	var rows int
	for i := 0; i < len(sets); i++ {
		for _, rowWeight := range sets[i].rowWeights {
			if weights == nil {
				merged.rowWeights = append(merged.rowWeights, rowWeight/float64(len(sets)))
			} else {
				merged.rowWeights = append(merged.rowWeights, rowWeight*weights[i])
			}
		}

		r, c := sets[i].scores.Dims()
		if c != len(columns) {
			panic("tnarg")
		}

		rows += r
	}

	merged.scores = mat.NewDense(rows, len(columns), nil)

	rout := 0
	for i := 0; i < len(sets); i++ {
		s := sets[i].scores
		r, _ := s.Dims()

		for j := 0; j < r; j++ {
			merged.rowNames = append(merged.rowNames, sets[i].rowNames[j])
			row := mat.Row(nil, j, s)
			merged.scores.SetRow(rout, row)
			rout++
		}
	}

	return &merged
}

// func CovarianceMatrix(dst *mat.SymDense, x mat.Matrix, weights []float64)
// func NewNormal(mu []float64, sigma mat.Symmetric, src rand.Source) (*Normal, bool)

func ScoreEducation(input Input) *ScoreSet {
	usnews := NewScoreSet("/Education/USNews", input.CandidateCities, BIGGER, func(city City) float64 {
		return float64(20000 - city.Education.USNews)
	})
	math := NewScoreSet("/Education/Math", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Education.MathProficiency
	})
	reading := NewScoreSet("/Education/Reading", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Education.ReadingProficiency
	})
	graduation := NewScoreSet("/Education/Graduation", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Education.GraduationRate
	})
	college := NewScoreSet("/Education/CollegeReadiness", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Education.CollegeReadiness
	})

	myscore := Merge([]*ScoreSet{
		math,
		reading,
		graduation,
		college,
	}, []float64{
		0.3,
		0.3,
		0.1,
		0.3,
	})

	return Merge([]*ScoreSet{
		usnews,
		myscore,
	}, nil)
}

func mortgage(P float64) float64 {
	apr := float64(3.20) // Mortgage rate
	Y := float64(30)     // Years

	n := math.Ceil(Y * 12) // Calculate payments from years
	// assuming monthly payments
	r := apr / 12
	return P * (r / 100 * math.Pow(1+r/100, n)) / (math.Pow(1+r/100, n) - 1)
}

func ScoreFinancial(input Input) *ScoreSet {
	return NewScoreSet("/Financial", input.CandidateCities, SMALLER, func(city City) float64 {
		loan := float64(city.RealEstate.MarketValue) - input.HomeEquity

		annual_income_tax := float64(input.AnnualIncome) * city.Taxes.Income
		annual_expenses := float64(input.AnnualExpenses) + (float64(input.AnnualExpenses) * city.Taxes.Sales)
		annual_property_taxes := float64(city.RealEstate.AssessedValue) * city.Taxes.Property

		monthly_payment := mortgage(loan)

		score := (12.0 * monthly_payment) +
			annual_income_tax +
			annual_expenses +
			annual_property_taxes

		return score
	})
}

func ScoreClimate(input Input) *ScoreSet {
	sunshine := NewScoreSet("/Livability/Climate/Sunshine", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Climate.SunnyDays
	})

	snow := NewScoreSet("/Livability/Climate/Snow", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Climate.SnowInches
	})

	precip := NewScoreSet("/Livability/Climate/Precip", input.CandidateCities, SMALLER, func(city City) float64 {
		score := USAverageRainfall - (city.Climate.RainInches + city.Climate.SnowInches)
		if score < 0.0 {
			score = -1.0 * score
		}
		return score
	})

	return Merge([]*ScoreSet{
		sunshine,
		precip,
		snow,
	}, []float64{
		0.45,
		0.45,
		0.10,
	})
}

func ScoreFamily(input Input) *ScoreSet {
	margaret := NewScoreSet("/Family/Margaret", input.CandidateCities, SMALLER, func(city City) float64 {
		return city.Family.MilesToMargaret
	})

	nich := NewScoreSet("/Family/Nich", input.CandidateCities, SMALLER, func(city City) float64 {
		return city.Family.MilesToNich
	})

	peggy := NewScoreSet("/Family/Peggy", input.CandidateCities, SMALLER, func(city City) float64 {
		return city.Family.MilesToPeggy
	})

	ryan := NewScoreSet("/Family/Ryan", input.CandidateCities, SMALLER, func(city City) float64 {
		return city.Family.MilesToRyan
	})

	return Merge([]*ScoreSet{
		margaret,
		nich,
		peggy,
		ryan,
	}, []float64{
		0.35,
		0.20,
		0.35,
		0.10,
	})
}

func ScoreLivability(input Input) *ScoreSet {
	crime := NewScoreSet("/Livability/Crime", input.CandidateCities, SMALLER, func(city City) float64 {
		return (0.3 * city.Crime.Violent) + (0.7 * city.Crime.Property)
	})

	politics := NewScoreSet("/Livability/Politics", input.CandidateCities, BIGGER, func(city City) float64 {
		return 1.0 - math.Abs(0.70-city.Livability.Politics)
	})

	climate := ScoreClimate(input)

	culture := NewScoreSet("/Livability/Culture", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Livability.Culture
	})

	running := NewScoreSet("/Livability/Running", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Livability.Running
	})

	walkscore := NewScoreSet("/Livability/WalkScore", input.CandidateCities, BIGGER, func(city City) float64 {
		return city.Livability.WalkScore
	})

	airport := NewScoreSet("/Livability/Airport", input.CandidateCities, SMALLER, func(city City) float64 {
		return city.Livability.MilesToAirport
	})

	return Merge([]*ScoreSet{
		Merge([]*ScoreSet{
			crime,
			politics,
			culture,
			running,
			climate,
			walkscore,
		}, nil),
		airport,
	}, []float64{
		0.95,
		0.05,
	})
}

func main() {
	data, err := ioutil.ReadFile("data.yaml")
	if err != nil {
		log.Fatal(err)
	}

	var input Input
	if err := yaml.Unmarshal(data, &input); err != nil {
		log.Fatal(err)
	}

	education := ScoreEducation(input)
	financial := ScoreFinancial(input)
	family := ScoreFamily(input)
	livability := ScoreLivability(input)

	merged := Merge([]*ScoreSet{
		education,
		family,
		financial,
		livability,
	}, []float64{
		0.33,
		0.33,
		0.17,
		0.17,
	})

	if false {
		panic(merged)
	}

	merged.Print()
	//PrintScores("Final", merged)
}
