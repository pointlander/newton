// Copyright 2022 The Newton Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	drw "image/draw"
	"image/gif"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"runtime"
	"time"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Eta is the learning rate
	EtaClassical = .001
	// Eta is the learning rate
	EtaQuantum = .1
	// AlphaQuantum is the momentum
	AlphaQuantum = .1
	// Epochs is the number of epochs
	Epochs = 512
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Softmax is the softmax function for big numbers
func Softmax(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	max := float32(0)
	for _, v := range a.X {
		if v > max {
			max = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := float64(max) * S
		sum := 0.0
		for j, ax := range a.X[i : i+width] {
			values[j] = math.Exp(float64(ax) - s)
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, float32(cx/sum))
		}
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

var (
	//FlagClassical classical mode
	FlagClassical = flag.Bool("classical", false, "classical mode")
	//FlagQuantum quantum mode
	FlagQuantum = flag.Bool("quantum", false, "quantum mode")
)

func main() {
	flag.Parse()
	if *FlagClassical {
		Classical()
		return
	}
	if *FlagQuantum {
		Quantum()
		return
	}
}

// Quantum is a quantum model
func Quantum() {
	rnd := rand.New(rand.NewSource(1))

	width, length := 8, 128

	// Create the weight data matrix
	set := tc128.NewSet()
	set.Add("particles", width, length)
	for _, w := range set.Weights {
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, complex((2*rnd.Float64()-1), (2*rnd.Float64()-1)))
		}
	}
	particles := set.ByName["particles"].X

	q := tc128.Mul(set.Get("particles"), set.Get("particles"))
	l1 := tc128.Mul(q, tc128.T(set.Get("particles")))
	l2 := tc128.Mul(tc128.H(q), l1)
	cost := tc128.Sum(tc128.Quadratic(set.Get("particles"), tc128.T(l2)))

	deltas := make([][]complex128, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]complex128, len(p.X)))
	}

	project := func(x []complex128) plotter.XYs {
		particles64 := make([]float64, 0, len(x))
		for _, v := range x {
			particles64 = append(particles64, float64(cmplx.Abs(v)))
		}
		ranks := mat.NewDense(length, width, particles64)
		var pc stat.PC
		ok := pc.PrincipalComponents(ranks, nil)
		if !ok {
			panic("PrincipalComponents failed")
		}
		k := 2
		var proj mat.Dense
		var vec mat.Dense
		pc.VectorsTo(&vec)
		proj.Mul(ranks, vec.Slice(0, width, 0, k))

		points := make(plotter.XYs, 0, 8)
		for i := 0; i < length; i++ {
			points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
		}
		return points
	}

	i, points, animation, states := 0, make(plotter.XYs, 0, 8), &gif.GIF{}, make([][]complex128, Epochs)
	// The stochastic gradient descent loop
	for i < Epochs {
		start := time.Now()
		// Calculate the gradients
		total := tc128.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		sum := complex128(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := cmplx.Sqrt(sum)
		scaling := complex128(1.0)
		if cmplx.Abs(norm) > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = AlphaQuantum*deltas[j][k] - EtaQuantum*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, cmplx.Abs(total), end)
		set.Zero()

		if math.IsNaN(cmplx.Abs(total)) {
			fmt.Println(total)
			break
		}

		state := make([]complex128, len(particles))
		copy(state, particles)
		states[i] = state

		points = append(points, plotter.XY{X: float64(i), Y: cmplx.Abs(total)})
		i++
	}

	done, images := make(chan bool, 8), make([]*image.Paletted, Epochs)
	generate := func(i int) {
		p := plot.New()

		p.Title.Text = "x vs y"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"

		scatter, err := plotter.NewScatter(project(states[i]))
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(3)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		c := vgimg.New(8*vg.Inch, 8*vg.Inch)
		p.Draw(draw.New(c))

		opts := gif.Options{
			NumColors: 256,
			Drawer:    drw.FloydSteinberg,
		}
		bounds := c.Image().Bounds()

		// More or less taken from the image/gif package
		paletted := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			paletted.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), c.Image())
		}
		opts.Drawer.Draw(paletted, bounds, c.Image(), image.Point{})

		images[i] = paletted
		done <- true
	}

	i, flight := 0, 0
	for i < runtime.NumCPU() && i < Epochs {
		go generate(i)
		flight++
		i++
	}

	for i < Epochs {
		<-done
		flight--
		go generate(i)
		flight++
		i++
	}

	for flight > 0 {
		<-done
		flight--
	}

	for _, paletted := range images {
		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 0)
	}

	f, _ := os.OpenFile("animation.gif", os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	for i := 0; i < length; i++ {
		for j := 0; j < width; j++ {
			fmt.Printf("%f ", cmplx.Abs(particles[i*width+j]))
		}
		fmt.Println()
	}

	p = plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err = plotter.NewScatter(project(particles))
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "projection.png")
	if err != nil {
		panic(err)
	}
}

// Classical is a classical model
func Classical() {
	// 4095 4083.6013 122.103734ms
	rnd := rand.New(rand.NewSource(1))

	width, length := 8, 2*1024
	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("points", width, length)
	for _, w := range set.Weights {
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64() - 1)))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	particles := set.ByName["points"].X

	dropout := tf32.U(func(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
		size, width := len(a.X), a.S[0]
		c, drops, factor := tf32.NewV(a.S...), make([]int, width), float32(1)/(1-.1)
		for i := range drops {
			if rnd.Float64() > .1 {
				drops[i] = 1
			}
		}
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
		if k(&c) {
			return true
		}
		for i := 0; i < size; i += width {
			for j := range a.D[i : i+width] {
				if drops[j] == 1 {
					a.D[i+j] += c.D[i+j]
				}
			}
		}
		return false
	})

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	l1 := softmax(tf32.Mul(set.Get("points"), set.Get("points")))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), dropout(l1)))
	cost := tf32.Sum(tf32.Entropy(l2))

	project := func(x []float32) plotter.XYs {
		particles64 := make([]float64, 0, len(x))
		for _, v := range x {
			particles64 = append(particles64, float64(v))
		}
		ranks := mat.NewDense(length, width, particles64)
		var pc stat.PC
		ok := pc.PrincipalComponents(ranks, nil)
		if !ok {
			panic("PrincipalComponents failed")
		}
		k := 2
		var proj mat.Dense
		var vec mat.Dense
		pc.VectorsTo(&vec)
		proj.Mul(ranks, vec.Slice(0, width, 0, k))

		points := make(plotter.XYs, 0, 8)
		for i := 0; i < length; i++ {
			points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
		}
		return points
	}

	points := make(plotter.XYs, 0, 8)
	animation := &gif.GIF{}
	// The stochastic gradient descent loop
	for i < 4*1024 {
		start := time.Now()
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)
		for j, w := range set.Weights {
			for k, d := range w.D {
				g := d
				m := B1*w.States[StateM][k] + (1-B1)*g
				v := B2*w.States[StateV][k] + (1-B2)*g*g
				w.States[StateM][k] = m
				w.States[StateV][k] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				set.Weights[j].X[k] -= EtaClassical * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, total, end)
		set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		i++

		p := plot.New()

		p.Title.Text = "x vs y"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"

		scatter, err := plotter.NewScatter(project(particles))
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(3)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		c := vgimg.New(8*vg.Inch, 8*vg.Inch)
		p.Draw(draw.New(c))

		opts := gif.Options{
			NumColors: 256,
			Drawer:    drw.FloydSteinberg,
		}
		bounds := c.Image().Bounds()

		// More or less taken from the image/gif package
		paletted := image.NewPaletted(bounds, palette.Plan9[:opts.NumColors])
		if opts.Quantizer != nil {
			paletted.Palette = opts.Quantizer.Quantize(make(color.Palette, 0, opts.NumColors), c.Image())
		}
		opts.Drawer.Draw(paletted, bounds, c.Image(), image.Point{})

		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 0)
	}

	f, _ := os.OpenFile("animation.gif", os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	gif.EncodeAll(f, animation)

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	for i := 0; i < length; i++ {
		for j := 0; j < width; j++ {
			fmt.Printf("%f ", particles[i*width+j])
		}
		fmt.Println()
	}

	p = plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err = plotter.NewScatter(project(particles))
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "projection.png")
	if err != nil {
		panic(err)
	}
}
