package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
)

type options struct {
	workload   string
	profileDir string
	repeats    int
	siftPath   string
	efSearch   int
	m          int
	efConst    int
}

func main() {
	var opts options
	flag.StringVar(
		&opts.workload,
		"workload",
		"all",
		"workload(s) to run: search, filtered, build, delete, vacuum, planner, all",
	)
	flag.StringVar(&opts.profileDir, "profile-dir", "", "directory for CPU and heap profiles")
	flag.IntVar(&opts.repeats, "repeats", 4, "repeat count for query-heavy workloads")
	flag.StringVar(
		&opts.siftPath,
		"sift",
		"",
		"optional path to sift10k_test.bin for the search workload",
	)
	flag.IntVar(&opts.efSearch, "ef-search", 200, "efSearch value for index")
	flag.IntVar(&opts.m, "m", 16, "M value for index")
	flag.IntVar(&opts.efConst, "ef-const", 200, "efConst value for index")
	flag.Parse()

	if opts.siftPath == "" && flag.NArg() > 0 {
		opts.siftPath = flag.Arg(0)
	}

	if err := run(opts); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(opts options) error {
	names, err := selectedWorkloads(opts.workload)
	if err != nil {
		return err
	}

	for i, name := range names {
		def, ok := workloadCatalog[name]
		if !ok {
			return fmt.Errorf("unknown workload %q", name)
		}

		if i > 0 {
			fmt.Println()
		}
		fmt.Printf("=== %s ===\n", name)

		if err := runWorkload(def, opts); err != nil {
			return err
		}
	}

	return nil
}

func runWorkload(def workloadDefinition, opts options) (err error) {
	ctx := &profileContext{opts: opts}
	if def.prepare != nil {
		if err := def.prepare(ctx); err != nil {
			if ctx.cleanup != nil {
				if cleanupErr := ctx.cleanup(); cleanupErr != nil {
					return fmt.Errorf("%w (cleanup error: %v)", err, cleanupErr)
				}
			}
			return err
		}
	}
	if ctx.cleanup != nil {
		defer func() {
			if cleanupErr := ctx.cleanup(); cleanupErr != nil && err == nil {
				err = cleanupErr
			}
		}()
	}

	return withProfiles(opts.profileDir, def.name, func() error {
		return def.run(ctx)
	})
}

func withProfiles(dir, name string, run func() error) (err error) {
	if dir == "" {
		return run()
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	cpuPath := filepath.Join(dir, name+".cpu.prof")
	cpuFile, err := os.Create(cpuPath)
	if err != nil {
		return err
	}

	started := false
	defer func() {
		if started {
			pprof.StopCPUProfile()
		}
		if closeErr := cpuFile.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}()

	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		return err
	}
	started = true

	err = run()
	pprof.StopCPUProfile()
	started = false

	runtime.GC()

	heapPath := filepath.Join(dir, name+".heap.prof")
	heapFile, heapErr := os.Create(heapPath)
	if heapErr != nil {
		if err == nil {
			err = heapErr
		}
		return err
	}
	defer func() {
		if closeErr := heapFile.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}()

	if writeErr := pprof.WriteHeapProfile(heapFile); writeErr != nil && err == nil {
		err = writeErr
	}
	return err
}

func selectedWorkloads(spec string) ([]string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "all" {
		out := make([]string, len(workloadOrder))
		copy(out, workloadOrder)
		return out, nil
	}

	parts := strings.Split(spec, ",")
	out := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		name := strings.TrimSpace(part)
		if name == "" {
			continue
		}
		if name == "all" {
			return selectedWorkloads("all")
		}
		if _, ok := workloadCatalog[name]; !ok {
			return nil, fmt.Errorf("unknown workload %q", name)
		}
		if _, dup := seen[name]; dup {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no workloads selected")
	}
	return out, nil
}
