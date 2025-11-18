package main

import (
	"crypto/md5"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type WorkItem struct {
	filePath string
	attempt  int
}

type ProcessedFile struct {
	path string
	hash string
}

type StateStore struct {
	mu        sync.RWMutex
	processed map[string]bool
	failed    map[string]int
}

func (s *StateStore) MarkProcessed(filePath string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.processed[filePath] = true
	delete(s.failed, filePath)
}

func (s *StateStore) MarkFailed(filePath string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failed[filePath]++
}

func (s *StateStore) IsProcessed(filePath string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.processed[filePath]
}

func (s *StateStore) GetFailureCount(filePath string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.failed[filePath]
}

func (s *StateStore) GetProcessedCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.processed)
}

func processFile(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

func worker(id int, workChan <-chan WorkItem, stateStore *StateStore, resultsChan chan<- ProcessedFile, wg *sync.WaitGroup) {
	defer wg.Done()

	for work := range workChan {
		if stateStore.IsProcessed(work.filePath) {
			continue
		}

		if work.attempt > 3 {
			log.Printf("Worker %d: Max retries exceeded for %s\n", id, work.filePath)
			stateStore.MarkProcessed(work.filePath)
			continue
		}

		hash, err := processFile(work.filePath)
		if err != nil {
			log.Printf("Worker %d: Error processing %s (attempt %d): %v\n", id, work.filePath, work.attempt, err)
			stateStore.MarkFailed(work.filePath)

			if work.attempt < 3 {
				go func(work WorkItem) {
					time.Sleep(time.Duration(work.attempt*100) * time.Millisecond)
					workChan <- WorkItem{filePath: work.filePath, attempt: work.attempt + 1}
				}(work)
			}
			continue
		}

		stateStore.MarkProcessed(work.filePath)
		resultsChan <- ProcessedFile{path: work.filePath, hash: hash}
	}
}

func generateTestFiles(count int, dir string) ([]string, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	files := make([]string, count)
	for i := 0; i < count; i++ {
		filePath := filepath.Join(dir, fmt.Sprintf("file_%d.txt", i))
		content := []byte(fmt.Sprintf("File %d content with some data %d\n", i, time.Now().UnixNano()))
		if err := os.WriteFile(filePath, content, 0644); err != nil {
			return nil, err
		}
		files[i] = filePath
	}
	return files, nil
}

func main() {
	const (
		maxGoroutines = 50
		fileCount     = 10000
		testDir       = "./test_files"
	)

	log.Println("Generating test files...")
	files, err := generateTestFiles(fileCount, testDir)
	if err != nil {
		log.Fatalf("Failed to generate test files: %v\n", err)
	}
	defer os.RemoveAll(testDir)

	log.Printf("Generated %d test files\n", len(files))

	stateStore := &StateStore{
		processed: make(map[string]bool),
		failed:    make(map[string]int),
	}

	workChan := make(chan WorkItem, maxGoroutines*2)
	resultsChan := make(chan ProcessedFile, maxGoroutines)

	var wg sync.WaitGroup

	for i := 0; i < maxGoroutines; i++ {
		wg.Add(1)
		go worker(i, workChan, stateStore, resultsChan, &wg)
	}

	go func() {
		for _, filePath := range files {
			workChan <- WorkItem{filePath: filePath, attempt: 0}
		}
	}()

	var resultWg sync.WaitGroup
	resultWg.Add(1)
	go func() {
		defer resultWg.Done()
		processedCount := 0
		for result := range resultsChan {
			processedCount++
			if processedCount%1000 == 0 {
				log.Printf("Processed %d files (hash: %s)\n", processedCount, result.hash[:8])
			}
		}
	}()

	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			count := stateStore.GetProcessedCount()
			log.Printf("Progress: %d/%d files processed (%.1f%%)\n", count, fileCount, float64(count)*100/float64(fileCount))
		}
	}()

	wg.Wait()
	close(workChan)
	close(resultsChan)
	resultWg.Wait()

	finalCount := stateStore.GetProcessedCount()
	log.Printf("Processing complete: %d/%d files processed\n", finalCount, fileCount)

	if finalCount != fileCount {
		log.Printf("Warning: Not all files were processed\n")
	}
}
