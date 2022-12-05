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
	"sync"
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
	// EtaDistributed is the learning rate
	EtaDistributed = .00001
	// Eta is the learning rate
	EtaQuantum = .001
	// AlphaQuantum is the momentum
	AlphaQuantum = .9
	// EpochsClassical is the number of epochs for classical mode
	EpochsClassical = 4 * 1024
	// Epochs is the number of epochs
	EpochsQuantum = 2 * 1024
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Message is a neural network message
type Message struct {
	I int
	V []float32
}

// Node is a node in the neural network
type Node struct {
	Index  int
	Width  int
	Length int
	Rnd    *rand.Rand
	In     []chan Message
	R      []chan Message
	Out    chan Message
	Reply  chan Message
	Set    tf32.Set
}

// NewNode creates a new node
func NewNode(seed int64, index, width, length int, in, R []chan Message) *Node {
	rnd := rand.New(rand.NewSource(seed))
	set := tf32.NewSet()
	set.Add("points", width, length)
	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64()-1)*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	return &Node{
		Index:  index,
		Width:  width,
		Length: length,
		Rnd:    rnd,
		In:     in,
		R:      R,
		Out:    make(chan Message, 8),
		Reply:  make(chan Message, 8),
		Set:    set,
	}
}

// Live brings the neural network to life
func (n *Node) Live(fire bool) {
	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}

	softmax := tf32.U(Softmax)
	a := tf32.Mul(n.Set.Get("points"), n.Set.Get("points"))
	l1 := softmax(a)
	l2 := softmax(tf32.Mul(tf32.T(n.Set.Get("points")), l1))
	cost := tf32.Sum(tf32.Entropy(l2))
	lock := sync.RWMutex{}
	go func() {
		for m := range n.Reply {
			weights := make([]float32, n.Width)
			lock.RLock()
			copy(weights, n.Set.Weights[0].X[m.I*n.Width:m.I*n.Width+n.Width])
			lock.RUnlock()
			m.V = weights
			fmt.Println("there", n.Index, m.I, m.V)
			n.Out <- m
		}
	}()
	for i := range n.In {
		go func(i int) {
			for m := range n.In[i] {
				fmt.Println("over there", n.Index, m.I, m.V, len(n.In[i]))
				lock.Lock()
				copy(n.Set.Weights[0].X[m.I*n.Width:m.I*n.Width+n.Width], m.V)
				lock.Unlock()
			}
		}(i)
	}
	for {
		// Calculate the gradients
		lock.RLock()
		total := tf32.Gradient(cost).X[0]
		lock.RUnlock()

		// Update the point weights with the partial derivatives using adam
		lock.Lock()
		b1, b2 := pow(B1), pow(B2)
		for j, w := range n.Set.Weights {
			for k, d := range w.D {
				g := d
				m := B1*w.States[StateM][k] + (1-B1)*g
				v := B2*w.States[StateV][k] + (1-B2)*g*g
				w.States[StateM][k] = m
				w.States[StateV][k] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				n.Set.Weights[j].X[k] -= EtaDistributed * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		lock.Unlock()

		n.Set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(n.Set.Weights[0].States)
			panic(fmt.Errorf("nan %d", n.Index))
		}

		i++

		if fire && i%(32*1024) == 0 {
			lock.RLock()
			max, x, y := float32(0.0), 0, 0
			a(func(a *tf32.V) bool {
				for i := 0; i < n.Length; i++ {
					for j := 0; j < n.Width; j++ {
						if a.X[i*n.Width+j] > max {
							max = a.X[i*n.Width+j]
							x, y = j, i
						}
					}
				}
				return true
			})
			fmt.Println(n.Index, x, y, max)
			n.R[x] <- Message{
				I: x,
			}
			n.R[y] <- Message{
				I: y,
			}

			average := make([]float32, n.Width)
			for j := 0; j < n.Length; j++ {
				for k := 0; k < n.Width; k++ {
					average[k] += n.Set.Weights[0].X[j*n.Width+k]
				}
			}
			lock.RUnlock()
			for k := 0; k < n.Width; k++ {
				average[k] /= float32(n.Length)
			}
			fmt.Println("output", n.Index, average)
			n.Out <- Message{
				I: n.Index,
				V: average,
			}
		}
	}
}

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
	// FlagClassical classical mode
	FlagClassical = flag.Bool("classical", false, "classical mode")
	// FlagQuantum quantum mode
	FlagQuantum = flag.Bool("quantum", false, "quantum mode")
	// FlagDistributed distributed mode
	FlagDistributed = flag.Bool("distributed", false, "distributed mode")
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
	if *FlagDistributed {
		Distributed()
		return
	}
}

// Distributed distributed mode
func Distributed() {
	in, reply, out, reply2 := make([]chan Message, 8), make([]chan Message, 8), make([]chan Message, 8), make([]chan Message, 8)
	for i := range in {
		in[i] = make(chan Message, 8)
		reply[i] = make(chan Message, 8)
	}
	nodes := make([]*Node, 8)
	for i := range nodes {
		nodes[i] = NewNode(int64(i+1), i, 8, 8, in, reply)
		out[i] = nodes[i].Out
		reply2[i] = nodes[i].Reply
	}
	head := NewNode(9, 9, 8, 8, out, reply2)
	for _, n := range nodes {
		go n.Live(false)
	}
	go head.Live(true)
	for m := range head.Out {
		fmt.Println(m)
	}
}

// Quantum is a quantum model
func Quantum() {
	rnd := rand.New(rand.NewSource(1))

	width, length := 8, 128

	// Create the weight data matrix
	set := tc128.NewSet()
	set.Add("particles", width, length)
	set.Add("w1", width, width)
	set.Add("w2", width, width)
	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, complex((2*rnd.Float64()-1)*factor, (2*rnd.Float64()-1)*factor))
		}
		w.States = make([][]complex128, 1)
		for i := range w.States {
			w.States[i] = make([]complex128, len(w.X))
		}
	}
	particles := set.ByName["particles"].X

	a := tc128.Mul(set.Get("w1"), set.Get("particles"))
	b := tc128.Mul(set.Get("w2"), set.Get("particles"))
	q := tc128.Mul(a, b)
	l1 := tc128.Mul(q, tc128.T(set.Get("particles")))
	l2 := tc128.Mul(tc128.H(q), l1)
	cost := tc128.Avg(tc128.Quadratic(set.Get("particles"), tc128.T(l2)))

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

	i, points, animation, states := 0, make(plotter.XYs, 0, 8), &gif.GIF{}, make([][]complex128, EpochsQuantum)
	// The stochastic gradient descent loop
	for i < EpochsQuantum {
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
				w.States[0][k] = AlphaQuantum*w.States[0][k] - EtaQuantum*d*scaling
				set.Weights[j].X[k] += w.States[0][k]
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

	done, images := make(chan bool, 8), make([]*image.Paletted, EpochsQuantum)
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
	for i < runtime.NumCPU() && i < EpochsQuantum {
		go generate(i)
		flight++
		i++
	}

	for i < EpochsQuantum {
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

	width, length := 8, 128
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
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64()-1)*factor))
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

	_ = dropout

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	l1 := softmax(tf32.Mul(set.Get("points"), set.Get("points")))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), l1))
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

	points, animation, states := make(plotter.XYs, 0, 8), &gif.GIF{}, make([][]float32, EpochsClassical)
	// The stochastic gradient descent loop
	for i < EpochsClassical+1 {
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

		state := make([]float32, len(particles))
		copy(state, particles)
		states[i-1] = state

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		i++
	}

	done, images := make(chan bool, 8), make([]*image.Paletted, EpochsClassical)
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
	for i < runtime.NumCPU() && i < EpochsClassical {
		go generate(i)
		flight++
		i++
	}

	for i < EpochsClassical {
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
