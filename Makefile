.PHONY: fmt vet test bench tidy check

fmt:
	@files="$$(git ls-files '*.go')"; \
	if [ -z "$$files" ]; then \
		echo "no tracked Go files to format"; \
	else \
		goimports -w $$files; \
		golines --base-formatter gofumpt -w $$files; \
	fi

vet:
	@go vet ./...

test:
	@go test -v -race ./...

bench:
	@go test -v -bench=. -benchmem ./...

build:
	@go build -v ./...

tidy:
	@go mod tidy

check: fmt vet test bench build
