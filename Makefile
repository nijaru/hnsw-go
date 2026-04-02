.PHONY: fmt fmt-check hooks vet test bench tidy check

fmt:
	@files="$$(git ls-files '*.go')"; \
	if [ -z "$$files" ]; then \
		echo "no tracked Go files to format"; \
	else \
		goimports -w $$files; \
		golines --base-formatter gofumpt -w $$files; \
	fi

fmt-check:
	@files="$$(git ls-files '*.go')"; \
	if [ -z "$$files" ]; then \
		echo "no tracked Go files to check"; \
	else \
		unformatted="$$( { goimports -l $$files; golines --base-formatter gofumpt -l $$files; } | sort -u )"; \
		if [ -n "$$unformatted" ]; then \
			echo "The following files are not formatted correctly:"; \
			printf '%s\n' "$$unformatted"; \
			echo "Please run 'make fmt' locally."; \
			exit 1; \
		fi; \
	fi

hooks:
	@git config core.hooksPath .githooks
	@echo "Configured git hooks path to .githooks"

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

check: fmt-check vet test bench build
