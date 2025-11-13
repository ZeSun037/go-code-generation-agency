package main

import (
	"fmt"
	"time"
)

// Metric represents a metric event
type Metric struct {
	Name  string
	Value int
}

// Collector collects metrics from multiple sources
type Collector struct {
	input chan Metric
}

// NewCollector creates a collector with a bounded buffer
func NewCollector(bufferSize int) *Collector {
	c := &Collector{
		input: make(chan Metric, bufferSize),
	}
	go c.run() // single goroutine processes all metrics
	return c
}

// run processes metrics sequentially
func (c *Collector) run() {
	storage := make(map[string]int)
	for m := range c.input {
		// Aggregate metric
		storage[m.Name] += m.Value
		fmt.Printf("Collected %s => %d\n", m.Name, storage[m.Name])
	}
}

// Push tries to send a metric to the collector
func (c *Collector) Push(m Metric) {
	// Non-blocking send: if channel full, drop metric (backpressure)
	select {
	case c.input <- m:
	default:
		fmt.Println("Dropped metric:", m.Name)
	}
}

func main() {
	collector := NewCollector(5) // small buffer to see backpressure

	// Simulate multiple sources sending metrics
	for i := 0; i < 10; i++ {
		collector.Push(Metric{Name: "requests", Value: 1})
		time.Sleep(100 * time.Millisecond)
	}

	time.Sleep(1 * time.Second) // wait for collector to process
}
