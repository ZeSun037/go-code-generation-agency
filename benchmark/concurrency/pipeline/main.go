package main

import (
	"fmt"
	"sync"
	"time"
)

// Stage 1: Generate numbers
func generate(nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _, n := range nums {
			out <- n
		}
	}()
	return out
}

// Stage 2: Square the numbers
func square(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			time.Sleep(time.Millisecond * 100) // Simulate work
			out <- n * n
		}
	}()
	return out
}

// Stage 3: Filter even numbers
func filterEven(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for n := range in {
			if n%2 == 0 {
				out <- n
			}
		}
	}()
	return out
}

// Fan-out: Split work across multiple goroutines
func fanOut(in <-chan int, n int, process func(<-chan int) <-chan int) []<-chan int {
	channels := make([]<-chan int, n)
	for i := 0; i < n; i++ {
		channels[i] = process(in)
	}
	return channels
}

// Fan-in: Merge multiple channels into one
func fanIn(channels ...<-chan int) <-chan int {
	out := make(chan int)
	var wg sync.WaitGroup

	for _, ch := range channels {
		wg.Add(1)
		go func(c <-chan int) {
			defer wg.Done()
			for n := range c {
				out <- n
			}
		}(ch)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func main() {
	fmt.Println("Starting concurrent pipeline...")

	// Create pipeline: generate -> fan-out square (3 workers) -> fan-in -> filter
	numbers := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

	// Fan-out square operation to 3 workers
	squaredChannels := fanOut(numbers, 3, square)

	// Fan-in results
	squared := fanIn(squaredChannels...)

	// Filter even numbers
	filtered := filterEven(squared)

	// Consume results
	fmt.Println("Even squared numbers:")
	for result := range filtered {
		fmt.Printf("%d ", result)
	}
	fmt.Println("\nPipeline completed!")
}
