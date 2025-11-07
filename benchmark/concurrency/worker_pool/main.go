package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// Job represents a task to be processed
type Job struct {
	ID    int
	Value int
}

// Result represents the output of a processed job
type Result struct {
	JobID  int
	Output int
}

// Worker processes jobs from the jobs channel
func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()

	for job := range jobs {
		fmt.Printf("Worker %d processing job %d\n", id, job.ID)

		// Simulate work with random sleep
		time.Sleep(time.Millisecond * time.Duration(rand.Intn(500)))

		// Process the job (squaring the value as an example)
		output := job.Value * job.Value

		results <- Result{
			JobID:  job.ID,
			Output: output,
		}
	}
}

func main() {
	const numJobs = 20
	const numWorkers = 4

	jobs := make(chan Job, numJobs)
	results := make(chan Result, numJobs)

	var wg sync.WaitGroup

	// Start workers
	for w := 1; w <= numWorkers; w++ {
		wg.Add(1)
		go worker(w, jobs, results, &wg)
	}

	// Send jobs
	go func() {
		for j := 1; j <= numJobs; j++ {
			jobs <- Job{ID: j, Value: j * 10}
		}
		close(jobs)
	}()

	// Close results channel after all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	totalProcessed := 0
	for result := range results {
		fmt.Printf("Job %d completed with output: %d\n", result.JobID, result.Output)
		totalProcessed++
	}

	fmt.Printf("\nProcessed %d jobs with %d workers\n", totalProcessed, numWorkers)
}
