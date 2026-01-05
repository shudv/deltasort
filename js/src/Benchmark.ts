import { User, generateSortedUsers, mutateUser, userComparator } from "./BenchmarkData";
import { deltaSort } from "./DeltaSort";
import * as fs from "fs";
import * as path from "path";

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/** Array size for main benchmarks */
const N = 50_000;

/** Delta counts to test */
const DELTA_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500];

/** Array sizes for crossover analysis */
const CROSSOVER_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000];

/** Number of iterations per benchmark (higher for JS due to JIT variance) */
const ITERATIONS = 200;

/** Number of iterations for crossover measurements */
const CROSSOVER_ITERATIONS = 20;

/** Z-score for 95% confidence interval */
const Z_95 = 1.96;

const shouldExport = process.argv.includes("--export");

// ============================================================================
// SORTING ALGORITHMS
// ============================================================================

function binaryInsertionSort<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
): T[] {
    const sortedDesc = Array.from(dirtyIndices).sort((a, b) => b - a);
    const values: T[] = [];

    for (const idx of sortedDesc) {
        values.push(arr[idx]!);
        arr.splice(idx, 1);
    }

    for (const value of values) {
        const insertIdx = binarySearchPosition(arr, value, cmp);
        arr.splice(insertIdx, 0, value);
    }

    return arr;
}

function extractSortMerge<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
): T[] {
    const sortedIndices = Array.from(dirtyIndices).sort((a, b) => b - a);
    const dirtyvalues: T[] = [];

    for (const idx of sortedIndices) {
        dirtyvalues.push(arr[idx]!);
        arr.splice(idx, 1);
    }

    dirtyvalues.sort(cmp);

    const result: T[] = [];
    let i = 0;
    let j = 0;

    while (i < arr.length && j < dirtyvalues.length) {
        if (cmp(arr[i]!, dirtyvalues[j]!) <= 0) {
            result.push(arr[i]!);
            i++;
        } else {
            result.push(dirtyvalues[j]!);
            j++;
        }
    }

    while (i < arr.length) {
        result.push(arr[i]!);
        i++;
    }

    while (j < dirtyvalues.length) {
        result.push(dirtyvalues[j]!);
        j++;
    }

    for (let k = 0; k < result.length; k++) {
        arr[k] = result[k]!;
    }

    return arr;
}

function binarySearchPosition<T>(arr: T[], value: T, cmp: (a: T, b: T) => number): number {
    let lo = 0;
    let hi = arr.length;

    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cmp(value, arr[mid]!) < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    return lo;
}

// ============================================================================
// STATISTICS
// ============================================================================

interface Stats {
    mean: number;
    ci95: number;
}

function calculateStats(values: number[]): Stats {
    const n = values.length;
    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1);
    const stdDev = Math.sqrt(variance);
    const stdError = stdDev / Math.sqrt(n);
    const ci95 = Z_95 * stdError;
    return { mean, ci95 };
}

// ============================================================================
// BENCHMARK INFRASTRUCTURE
// ============================================================================

interface BenchmarkResult {
    timeUs: number;
    timeCi: number;
    comparisons: number;
    comparisonsCi: number;
}

interface AlgorithmResult {
    k: number;
    timeUs: number;
    timeCi: number;
    comparisons: number;
    comparisonsCi: number;
}

interface BenchmarkResults {
    native: AlgorithmResult[];
    bis: AlgorithmResult[];
    esm: AlgorithmResult[];
    deltaSort: AlgorithmResult[];
}

interface CrossoverResult {
    n: number;
    crossoverRatio: number;
}

function sampleDistinctIndices(n: number, k: number): number[] {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
        const j = i + Math.floor(Math.random() * (n - i));
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr.slice(0, k);
}

/**
 * Run benchmark with separate timing and counting phases.
 * Each iteration generates fresh random mutations for proper variance measurement.
 * Phase 1: Measure time with base comparator (no counting overhead)
 * Phase 2: Measure comparison counts (separate runs to get variance)
 */
function runBenchmark(
    baseUsers: User[],
    k: number,
    sortFn: (arr: User[], cmp: (a: User, b: User) => number, dirty: Set<number>) => void,
): BenchmarkResult {
    const n = baseUsers.length;

    // Phase 1: Timing (no counting overhead, fresh mutations each iteration)
    const times: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        const dirtyIndices = new Set<number>();
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }
        const start = performance.now();
        sortFn(users, userComparator, dirtyIndices);
        times.push((performance.now() - start) * 1000); // µs
    }

    // Phase 2: Comparison counts (fresh mutations each iteration)
    const compCounts: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        const dirtyIndices = new Set<number>();
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }
        const counter = { count: 0 };
        const countingCmp = (a: User, b: User): number => {
            counter.count++;
            return userComparator(a, b);
        };
        sortFn(users, countingCmp, dirtyIndices);
        compCounts.push(counter.count);
    }

    const timeStats = calculateStats(times);
    const compStats = calculateStats(compCounts);

    return {
        timeUs: timeStats.mean,
        timeCi: timeStats.ci95,
        comparisons: compStats.mean,
        comparisonsCi: compStats.ci95,
    };
}

function runNativeBenchmark(baseUsers: User[], k: number): BenchmarkResult {
    const n = baseUsers.length;

    // Phase 1: Timing with fresh mutations each iteration
    const times: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
        }
        const start = performance.now();
        users.sort(userComparator);
        times.push((performance.now() - start) * 1000);
    }

    // Phase 2: Comparison counts with fresh mutations
    const compCounts: number[] = [];
    for (let i = 0; i < ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
        }
        const counter = { count: 0 };
        const countingCmp = (a: User, b: User): number => {
            counter.count++;
            return userComparator(a, b);
        };
        users.sort(countingCmp);
        compCounts.push(counter.count);
    }

    const timeStats = calculateStats(times);
    const compStats = calculateStats(compCounts);

    return {
        timeUs: timeStats.mean,
        timeCi: timeStats.ci95,
        comparisons: compStats.mean,
        comparisonsCi: compStats.ci95,
    };
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

function measureTime(fn: () => void): number {
    const start = performance.now();
    fn();
    return (performance.now() - start) * 1000;
}

function deltaSortIsFaster(baseUsers: User[], k: number, n: number): boolean {
    let nativeTime = 0;
    let dsTime = 0;

    for (let i = 0; i < CROSSOVER_ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const dirtyIndices = new Set<number>();

        for (let j = 0; j < k; j++) {
            const idx = Math.floor(Math.random() * n);
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }

        const usersForNative = users.map((u) => ({ ...u }));
        const usersForDs = users.map((u) => ({ ...u }));

        nativeTime += measureTime(() => usersForNative.sort(userComparator));
        dsTime += measureTime(() => deltaSort(usersForDs, dirtyIndices, userComparator));
    }

    return dsTime < nativeTime;
}

function findCrossover(n: number): number {
    const baseUsers = generateSortedUsers(n);

    // Warmup
    for (let i = 0; i < 5; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        users.sort(userComparator);
    }

    let lo = 1;
    let hi = Math.floor((n * 2) / 5);

    if (!deltaSortIsFaster(baseUsers, 1, n)) {
        return 0;
    }

    if (deltaSortIsFaster(baseUsers, n, n)) {
        return n;
    }

    const minRange = Math.floor(n * 0.001);

    while (lo < hi) {
        if (hi - lo < minRange) {
            break;
        }

        const mid = lo + Math.floor((hi - lo + 1) / 2);

        if (deltaSortIsFaster(baseUsers, mid, n)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    return lo;
}

// ============================================================================
// OUTPUT
// ============================================================================

function formatNumber(n: number): string {
    if (n >= 1_000_000) return `${n / 1_000_000}M`;
    if (n >= 1_000) return `${n / 1_000}K`;
    return `${n}`;
}

/** Format value ± ci with consistent total width */
function formatWithCi(value: number, ci: number, totalWidth: number): string {
    const valStr = value.toFixed(1);
    const ciStr = ci.toFixed(1);
    const content = `${valStr} ± ${ciStr}`;
    return content.padStart(totalWidth);
}

/** Format integer value ± ci with consistent total width */
function formatIntWithCi(value: number, ci: number, totalWidth: number): string {
    const valStr = Math.round(value).toString();
    const ciStr = Math.round(ci).toString();
    const content = `${valStr} ± ${ciStr}`;
    return content.padStart(totalWidth);
}

function printExecutionTimeTable(results: BenchmarkResults): void {
    const COL_WIDTH = 17;

    console.log();
    console.log(`Execution Time (µs) - n=${formatNumber(N)}`);
    console.log(
        "┌────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐",
    );
    console.log(
        "│   k    │      Native       │        BIS        │        ESM        │     DeltaSort     │",
    );
    console.log(
        "├────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤",
    );
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltaSort[i]!;
        console.log(
            `│ ${n.k.toString().padStart(6)} │ ${formatWithCi(n.timeUs, n.timeCi, COL_WIDTH)} │ ${formatWithCi(b.timeUs, b.timeCi, COL_WIDTH)} │ ${formatWithCi(e.timeUs, e.timeCi, COL_WIDTH)} │ ${formatWithCi(d.timeUs, d.timeCi, COL_WIDTH)} │`,
        );
    }
    console.log(
        "└────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘",
    );
}

function printComparatorCountTable(results: BenchmarkResults): void {
    const COL_WIDTH = 17;

    console.log();
    console.log(`Comparator Invocations - n=${formatNumber(N)}`);
    console.log(
        "┌────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐",
    );
    console.log(
        "│   k    │      Native       │        BIS        │        ESM        │     DeltaSort     │",
    );
    console.log(
        "├────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤",
    );
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltaSort[i]!;
        console.log(
            `│ ${n.k.toString().padStart(6)} │ ${formatIntWithCi(n.comparisons, n.comparisonsCi, COL_WIDTH)} │ ${formatIntWithCi(b.comparisons, b.comparisonsCi, COL_WIDTH)} │ ${formatIntWithCi(e.comparisons, e.comparisonsCi, COL_WIDTH)} │ ${formatIntWithCi(d.comparisons, d.comparisonsCi, COL_WIDTH)} │`,
        );
    }
    console.log(
        "└────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘",
    );
}

function printCrossoverTable(results: CrossoverResult[]): void {
    console.log();
    console.log("Crossover Threshold");
    console.log("┌────────────┬──────────────┐");
    console.log("│     n      │  k_c/n (%)   │");
    console.log("├────────────┼──────────────┤");
    for (const r of results) {
        console.log(
            `│ ${formatNumber(r.n).padStart(10)} │ ${r.crossoverRatio.toFixed(1).padStart(11)}% │`,
        );
    }
    console.log("└────────────┴──────────────┘");
}

// ============================================================================
// CSV EXPORT (values only, no CI)
// ============================================================================

function exportExecutionTimeCsv(results: BenchmarkResults, filePath: string): void {
    let csv = "k,native,bis,esm,deltaSort\n";
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltaSort[i]!;
        csv += `${n.k},${n.timeUs.toFixed(1)},${b.timeUs.toFixed(1)},${e.timeUs.toFixed(1)},${d.timeUs.toFixed(1)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

function exportComparatorCountCsv(results: BenchmarkResults, filePath: string): void {
    let csv = "k,native,bis,esm,deltaSort\n";
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltaSort[i]!;
        csv += `${n.k},${Math.round(n.comparisons)},${Math.round(b.comparisons)},${Math.round(e.comparisons)},${Math.round(d.comparisons)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

function exportCrossoverCsv(results: CrossoverResult[], filePath: string): void {
    let csv = "n,crossover_ratio\n";
    for (const r of results) {
        csv += `${r.n},${r.crossoverRatio.toFixed(1)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

// ============================================================================
// MAIN
// ============================================================================

async function main(): Promise<void> {
    console.log();
    console.log("DeltaSort Benchmark");
    console.log("===================");

    const baseUsers = generateSortedUsers(N);

    // Warmup
    for (let i = 0; i < 5; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        users.sort(userComparator);
    }

    // --- Combined Execution Time & Comparator Count ---
    console.log();
    console.log("Running benchmarks (time + comparisons)...");
    const results: BenchmarkResults = {
        native: [],
        bis: [],
        esm: [],
        deltaSort: [],
    };

    for (const k of DELTA_COUNTS) {
        process.stdout.write(`  k=${k.toString().padStart(5)}...`);

        const native = runNativeBenchmark(baseUsers, k);
        results.native.push({ k, ...native });

        const bis = runBenchmark(baseUsers, k, (arr, cmp, dirty) =>
            binaryInsertionSort(arr, cmp, dirty),
        );
        results.bis.push({ k, ...bis });

        const esm = runBenchmark(baseUsers, k, (arr, cmp, dirty) =>
            extractSortMerge(arr, cmp, dirty),
        );
        results.esm.push({ k, ...esm });

        const ds = runBenchmark(baseUsers, k, (arr, cmp, dirty) => deltaSort(arr, dirty, cmp));
        results.deltaSort.push({ k, ...ds });

        console.log(" done");
    }

    printExecutionTimeTable(results);
    printComparatorCountTable(results);

    // --- Crossover Analysis ---
    console.log();
    console.log("Running crossover analysis (this may take a while)...");
    const crossoverResults: CrossoverResult[] = [];

    for (const size of CROSSOVER_SIZES) {
        process.stdout.write(`  n=${formatNumber(size).padStart(10)}...`);
        const kc = findCrossover(size);
        const crossoverRatio = (kc / size) * 100;
        crossoverResults.push({ n: size, crossoverRatio });
        console.log(` k_c=${kc} (${crossoverRatio.toFixed(1)}%)`);
    }

    printCrossoverTable(crossoverResults);

    // --- Export CSVs ---
    if (shouldExport) {
        console.log();
        console.log("Exporting CSV files...");
        const currentDir = path.dirname(new URL(import.meta.url).pathname);
        const basePath = path.join(currentDir, "../../paper/benchmarks/js");
        fs.mkdirSync(basePath, { recursive: true });
        exportExecutionTimeCsv(results, path.join(basePath, "execution-time.csv"));
        exportComparatorCountCsv(results, path.join(basePath, "comparator-count.csv"));
        exportCrossoverCsv(crossoverResults, path.join(basePath, "crossover-threshold.csv"));
    }

    console.log();
    console.log("Done!");
    if (!shouldExport) {
        console.log("Run with --export to write CSV files to paper/benchmarks/js/");
    }
}

main().catch(console.error);
