package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type WebScraper struct {
	client      *http.Client
	rateLimiter *time.Ticker
	maxWorkers  int
	semaphore   chan struct{}
	mu          sync.Mutex
	results     map[string]string
	errors      map[string]error
}

func NewWebScraper(requestsPerSecond float64, maxWorkers int) *WebScraper {
	interval := time.Duration(float64(time.Second) / requestsPerSecond)
	return &WebScraper{
		client:      &http.Client{Timeout: 10 * time.Second},
		rateLimiter: time.NewTicker(interval),
		maxWorkers:  maxWorkers,
		semaphore:   make(chan struct{}, maxWorkers),
		results:     make(map[string]string),
		errors:      make(map[string]error),
	}
}

func (ws *WebScraper) Close() {
	ws.rateLimiter.Stop()
}

func (ws *WebScraper) FetchURL(ctx context.Context, url string) (string, error) {
	select {
	case <-ws.rateLimiter.C:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	select {
	case ws.semaphore <- struct{}{}:
	case <-ctx.Done():
		return "", ctx.Err()
	}
	defer func() { <-ws.semaphore }()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request for %s: %w", url, err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

	resp, err := ws.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to fetch %s: %w", url, err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fmt.Printf("warning: failed to close response body: %v\n", err)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("got status code %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body for %s: %w", url, err)
	}

	return string(body), nil
}

func (ws *WebScraper) ScrapeURLs(ctx context.Context, urls []string) error {
	var wg sync.WaitGroup

	for _, url := range urls {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()

			content, err := ws.FetchURL(ctx, u)
			ws.mu.Lock()
			if err != nil {
				ws.errors[u] = err
			} else {
				ws.results[u] = content
			}
			ws.mu.Unlock()
		}(url)
	}

	wg.Wait()
	return nil
}

func (ws *WebScraper) GetResults() (map[string]string, map[string]error) {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	resultsCopy := make(map[string]string, len(ws.results))
	for k, v := range ws.results {
		resultsCopy[k] = v
	}

	errorsCopy := make(map[string]error, len(ws.errors))
	for k, v := range ws.errors {
		errorsCopy[k] = v
	}

	return resultsCopy, errorsCopy
}

func extractTextFromHTML(html string) string {
	text := strings.ReplaceAll(html, "<", "")
	text = strings.ReplaceAll(text, ">", "")
	text = strings.ReplaceAll(text, "<!--", "")
	text = strings.ReplaceAll(text, "-->", "")

	textLines := strings.Split(text, "\n")
	relevantLines := make([]string, 0)

	for _, line := range textLines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 20 && len(trimmed) < 500 {
			relevantLines = append(relevantLines, trimmed)
		}
	}

	if len(relevantLines) > 5 {
		relevantLines = relevantLines[:5]
	}

	summary := strings.Join(relevantLines, " ")
	if len(summary) > 200 {
		summary = summary[:200]
	}

	return summary
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	scraper := NewWebScraper(2.0, 3)
	defer scraper.Close()

	urls := []string{
		"https://www.example.com",
		"https://www.wikipedia.org",
		"https://www.github.com",
	}

	fmt.Println("Starting concurrent web scraper with rate limiting...")
	fmt.Printf("Scraping %d URLs with 2 requests/second and 3 concurrent workers\n", len(urls))

	if err := scraper.ScrapeURLs(ctx, urls); err != nil {
		fmt.Printf("Error during scraping: %v\n", err)
		return
	}

	results, errors := scraper.GetResults()

	fmt.Println("=== Scraping Results ===")
	for url, content := range results {
		fmt.Printf("URL: %s\n", url)
		fmt.Printf("Content length: %d bytes\n", len(content))

		summary := extractTextFromHTML(content)
		fmt.Printf("Extracted Summary: %s\n", summary)
		fmt.Println()
	}

	if len(errors) > 0 {
		fmt.Println("=== Errors ===")
		for url, err := range errors {
			fmt.Printf("URL: %s\n", url)
			fmt.Printf("Error: %v\n", err)
			fmt.Println()
		}
	}

	fmt.Printf("Successfully scraped: %d URLs\n", len(results))
	fmt.Printf("Failed: %d URLs\n", len(errors))
}
