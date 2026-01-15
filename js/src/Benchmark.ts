/**
 * DeltaSort Benchmark Suite (JavaScript)
 *
 * Run with: `pnpm benchmark`
 * Export CSV: `pnpm benchmark:export`
 */

import { User, generateSortedUsers, mutateUser, userComparator } from "./BenchmarkData";
import { deltaSort } from "./DeltaSort";
import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/** Array size for main benchmarks */
const N = 100_000;

/** Base number of iterations per benchmark (scaled up for small k) */
const BASE_ITERATIONS = 20;

/** Delta counts to test */
const DELTA_COUNTS = [
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
];

/** Number of iterations for crossover measurements */
const CROSSOVER_ITERATIONS = 10;

/** Array sizes for crossover analysis */
const CROSSOVER_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]; // 200_000, 500_000, 1_000_000];

/** Z-score for 95% confidence interval */
const Z_95 = 1.96;

const shouldExport = process.argv.includes("--export");

/** Get number of iterations for a given k value */
function iterationsForK(k: number): number {
    if (k <= 10) return BASE_ITERATIONS * 10;
    if (k <= 50) return BASE_ITERATIONS * 5;
    if (k <= 200) return BASE_ITERATIONS * 2;
    return BASE_ITERATIONS;
}

// ============================================================================
// SORTING ALGORITHMS
// ============================================================================

type Comparator<T> = (a: T, b: T) => number;

function binaryInsertionSort<T>(arr: T[], dirtyIndices: Set<number>, cmp: Comparator<T>): T[] {
    if (dirtyIndices.size === 0) return arr;

    // Sort indices descending for back-to-front extraction
    const sortedDesc = Array.from(dirtyIndices).sort((a, b) => b - a);
    const extracted: T[] = [];

    // Extract dirty values from back to front
    for (const idx of sortedDesc) {
        extracted.push(arr[idx]!);
        arr.splice(idx, 1);
    }

    // Binary insert each dirty value
    for (const value of extracted) {
        let lo = 0;
        let hi = arr.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (cmp(arr[mid]!, value) < 0) lo = mid + 1;
            else hi = mid;
        }
        arr.splice(lo, 0, value);
    }

    return arr;
}

function extractSortMerge<T>(arr: T[], dirtyIndices: Set<number>, cmp: Comparator<T>): T[] {
    if (dirtyIndices.size === 0) return arr;

    const n = arr.length;
    const k = dirtyIndices.size;

    // Sort indices ascending for single-pass extraction
    const sortedIndices = Array.from(dirtyIndices).sort((a, b) => a - b);

    // Single O(n) pass with index comparison
    const cleanValues: T[] = [];
    const dirtyValues: T[] = [];
    let dirtyPtr = 0;

    for (let i = 0; i < n; i++) {
        if (dirtyPtr < k && sortedIndices[dirtyPtr] === i) {
            dirtyValues.push(arr[i]!);
            dirtyPtr++;
        } else {
            cleanValues.push(arr[i]!);
        }
    }

    // Sort dirty values
    dirtyValues.sort(cmp);

    // Merge
    const result: T[] = [];
    let i = 0;
    let j = 0;

    while (i < cleanValues.length && j < dirtyValues.length) {
        if (cmp(cleanValues[i]!, dirtyValues[j]!) <= 0) {
            result.push(cleanValues[i]!);
            i++;
        } else {
            result.push(dirtyValues[j]!);
            j++;
        }
    }

    while (i < cleanValues.length) result.push(cleanValues[i++]!);
    while (j < dirtyValues.length) result.push(dirtyValues[j++]!);

    // Copy back
    for (let idx = 0; idx < n; idx++) arr[idx] = result[idx]!;

    return result;
}

// ============================================================================
// STATISTICS
// ============================================================================

interface Stats {
    mean: number;
    sd: number;
    ci95: number;
    cv: number;
}

function calculateStats(values: number[]): Stats {
    const n = values.length;
    if (n < 2) {
        const mean = n > 0 ? values[0]! : 0;
        return { mean, sd: 0, ci95: 0, cv: 0 };
    }
    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1);
    const sd = Math.sqrt(variance);
    const stdError = sd / Math.sqrt(n);
    const ci95 = Z_95 * stdError;
    const cv = mean > 0 ? (sd / mean) * 100 : 0;
    return { mean, sd, ci95, cv };
}

// ============================================================================
// BENCHMARK INFRASTRUCTURE
// ============================================================================

interface BenchmarkResult {
    timeUs: number;
    timeSd: number;
    timeCi: number;
    timeCv: number;
    comparisons: number;
    comparisonsSd: number;
    comparisonsCi: number;
    comparisonsCv: number;
    iterations: number;
}

interface AlgorithmResult {
    k: number;
    iterations: number;
    timeUs: number;
    timeSd: number;
    timeCi: number;
    timeCv: number;
    comparisons: number;
    comparisonsSd: number;
    comparisonsCi: number;
    comparisonsCv: number;
}

interface BenchmarkResults {
    native: AlgorithmResult[];
    bis: AlgorithmResult[];
    esm: AlgorithmResult[];
    deltasort: AlgorithmResult[];
}

interface CrossoverResultsAll {
    n: number;
    bisKc: number;
    bisRatio: number;
    esmKc: number;
    esmRatio: number;
    deltasortKc: number;
    deltasortRatio: number;
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
 */
function runBenchmark(
    baseUsers: User[],
    k: number,
    sortFn: (arr: User[], dirty: Set<number>, cmp: Comparator<User>) => void,
): BenchmarkResult {
    const n = baseUsers.length;
    const iters = iterationsForK(k);

    // Phase 1: Timing (no counting overhead)
    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        const dirtyIndices = new Set<number>();
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }
        const start = performance.now();
        sortFn(users, dirtyIndices, userComparator);
        times.push((performance.now() - start) * 1000); // µs
    }

    // Phase 2: Comparison counts (10 iterations for variance)
    const compCounts: number[] = [];
    for (let i = 0; i < 10; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        const dirtyIndices = new Set<number>();
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }
        let count = 0;
        const countingCmp = (a: User, b: User): number => {
            count++;
            return userComparator(a, b);
        };
        sortFn(users, dirtyIndices, countingCmp);
        compCounts.push(count);
    }

    const timeStats = calculateStats(times);
    const compStats = calculateStats(compCounts);

    return {
        timeUs: timeStats.mean,
        timeSd: timeStats.sd,
        timeCi: timeStats.ci95,
        timeCv: timeStats.cv,
        comparisons: compStats.mean,
        comparisonsSd: compStats.sd,
        comparisonsCi: compStats.ci95,
        comparisonsCv: compStats.cv,
        iterations: iters,
    };
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

function algorithmIsFaster(
    baseUsers: User[],
    k: number,
    n: number,
    algo: (arr: User[], dirty: Set<number>) => void,
): boolean {
    let nativeTime = 0;
    let algoTime = 0;

    for (let i = 0; i < CROSSOVER_ITERATIONS; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        const indices = sampleDistinctIndices(n, k);
        const dirtyIndices = new Set<number>();
        for (const idx of indices) {
            users[idx] = mutateUser(users[idx]!);
            dirtyIndices.add(idx);
        }

        const usersNative = users.map((u) => ({ ...u }));
        const usersAlgo = users.map((u) => ({ ...u }));

        const startNative = performance.now();
        usersNative.sort(userComparator);
        nativeTime += performance.now() - startNative;

        const startAlgo = performance.now();
        algo(usersAlgo, dirtyIndices);
        algoTime += performance.now() - startAlgo;
    }

    return algoTime < nativeTime;
}

function deltasortIsFaster(baseUsers: User[], k: number, n: number): boolean {
    return algorithmIsFaster(baseUsers, k, n, (arr, indices) =>
        deltaSort(arr, indices, userComparator),
    );
}

function bisIsFaster(baseUsers: User[], k: number, n: number): boolean {
    return algorithmIsFaster(baseUsers, k, n, (arr, indices) =>
        binaryInsertionSort(arr, indices, userComparator),
    );
}

function esmIsFaster(baseUsers: User[], k: number, n: number): boolean {
    return algorithmIsFaster(baseUsers, k, n, (arr, indices) =>
        extractSortMerge(arr, indices, userComparator),
    );
}

function findCrossoverGeneric(
    n: number,
    loRatio: number,
    hiRatio: number,
    isFaster: (baseUsers: User[], k: number, n: number) => boolean,
): number {
    const baseUsers = generateSortedUsers(n);

    // Warmup
    for (let i = 0; i < 5; i++) {
        const users = baseUsers.map((u) => ({ ...u }));
        users.sort(userComparator);
    }

    let lo = Math.max(1, Math.floor(n * loRatio));
    let hi = Math.floor(n * hiRatio);

    while (lo < hi) {
        const mid = lo + Math.ceil((hi - lo) / 2);
        if (isFaster(baseUsers, mid, n)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    return lo;
}

function findCrossover(n: number): number {
    return findCrossoverGeneric(n, 0.0, 0.5, deltasortIsFaster);
}

function findCrossoverBis(n: number): number {
    return findCrossoverGeneric(n, 0.0, 1.0, bisIsFaster);
}

function findCrossoverEsm(n: number): number {
    return findCrossoverGeneric(n, 0.0, 1.0, esmIsFaster);
}

// ============================================================================
// OUTPUT
// ============================================================================

function formatNumber(n: number): string {
    if (n >= 1_000_000) return `${n / 1_000_000}M`;
    if (n >= 1_000) return `${n / 1_000}K`;
    return `${n}`;
}

function formatWithCi(value: number, ci: number, totalWidth: number): string {
    const valStr = value.toFixed(1);
    const ciPercent = value > 0 ? ((ci / value) * 100).toFixed(1) : "0.0";
    const content = `${valStr} ±${ciPercent}%`;
    return content.padStart(totalWidth);
}

function formatIntWithCi(value: number, ci: number, totalWidth: number): string {
    const valStr = Math.round(value).toString();
    const ciPercent = value > 0 ? ((ci / value) * 100).toFixed(1) : "0.0";
    const content = `${valStr} ±${ciPercent}%`;
    return content.padStart(totalWidth);
}

function printExecutionTimeTable(results: BenchmarkResults): void {
    const COL_WIDTH = 15;

    console.log();
    console.log(`Execution Time (µs) - n=${formatNumber(N)}`);
    console.log(
        "┌────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐",
    );
    console.log(
        "│   k    │     Native      │       BIS       │       ESM       │    DeltaSort    │",
    );
    console.log(
        "├────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤",
    );
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltasort[i]!;
        console.log(
            `│ ${n.k.toString().padStart(6)} │ ${formatWithCi(n.timeUs, n.timeCi, COL_WIDTH)} │ ${formatWithCi(b.timeUs, b.timeCi, COL_WIDTH)} │ ${formatWithCi(e.timeUs, e.timeCi, COL_WIDTH)} │ ${formatWithCi(d.timeUs, d.timeCi, COL_WIDTH)} │`,
        );
    }
    console.log(
        "└────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘",
    );
}

function printCrossoverTableAll(results: CrossoverResultsAll[]): void {
    console.log();
    console.log("Crossover Threshold (All Algorithms vs Native)");
    console.log(
        "┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐",
    );
    console.log(
        "│     n      │  BIS k_c   │  BIS k_c%  │  ESM k_c   │  ESM k_c%  │   DS k_c   │  DS k_c%   │",
    );
    console.log(
        "├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤",
    );
    for (const r of results) {
        console.log(
            `│ ${formatNumber(r.n).padStart(10)} │ ${formatNumber(r.bisKc).padStart(10)} │ ${r.bisRatio.toFixed(3).padStart(9)}% │ ${formatNumber(r.esmKc).padStart(10)} │ ${r.esmRatio.toFixed(3).padStart(9)}% │ ${formatNumber(r.deltasortKc).padStart(10)} │ ${r.deltasortRatio.toFixed(3).padStart(9)}% │`,
        );
    }
    console.log(
        "└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘",
    );
}

// ============================================================================
// CSV EXPORT (with full statistics: mean, SD, CI, CV, iterations)
// ============================================================================

function exportExecutionTimeCsv(results: BenchmarkResults, filePath: string): void {
    let csv =
        "k,iters,native,native_sd,native_ci,native_cv,bis,bis_sd,bis_ci,bis_cv,esm,esm_sd,esm_ci,esm_cv,deltasort,deltasort_sd,deltasort_ci,deltasort_cv\n";
    for (let i = 0; i < results.native.length; i++) {
        const n = results.native[i]!;
        const b = results.bis[i]!;
        const e = results.esm[i]!;
        const d = results.deltasort[i]!;
        csv += `${n.k},${n.iterations},${n.timeUs.toFixed(1)},${n.timeSd.toFixed(1)},${n.timeCi.toFixed(1)},${n.timeCv.toFixed(1)},${b.timeUs.toFixed(1)},${b.timeSd.toFixed(1)},${b.timeCi.toFixed(1)},${b.timeCv.toFixed(1)},${e.timeUs.toFixed(1)},${e.timeSd.toFixed(1)},${e.timeCi.toFixed(1)},${e.timeCv.toFixed(1)},${d.timeUs.toFixed(1)},${d.timeSd.toFixed(1)},${d.timeCi.toFixed(1)},${d.timeCv.toFixed(1)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

function exportCrossoverAllCsv(results: CrossoverResultsAll[], filePath: string): void {
    let csv = "n,bis_kc,bis,esm_kc,esm,deltasort_kc,deltasort\n";
    for (const r of results) {
        csv += `${r.n},${r.bisKc},${r.bisRatio.toFixed(3)},${r.esmKc},${r.esmRatio.toFixed(3)},${r.deltasortKc},${r.deltasortRatio.toFixed(3)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

function exportMetadataCsv(results: BenchmarkResults, filePath: string): void {
    const timestamp = Date.now();
    const platform = `${process.platform}/${process.arch}`;

    // Try to get CPU info on macOS
    let machine = platform;
    try {
        if (process.platform === "darwin") {
            machine = execSync("sysctl -n machdep.cpu.brand_string", { encoding: "utf8" }).trim();
        }
    } catch {
        // Ignore errors
    }

    // Compute max CI as percentage of mean across all timing measurements
    // This represents our least confident measurement (can be narrowed with more iterations)
    let maxCiPercent = 0;
    for (const r of results.native) {
        if (r.timeUs > 0) maxCiPercent = Math.max(maxCiPercent, (r.timeCi / r.timeUs) * 100);
    }
    for (const r of results.bis) {
        if (r.timeUs > 0) maxCiPercent = Math.max(maxCiPercent, (r.timeCi / r.timeUs) * 100);
    }
    for (const r of results.esm) {
        if (r.timeUs > 0) maxCiPercent = Math.max(maxCiPercent, (r.timeCi / r.timeUs) * 100);
    }
    for (const r of results.deltasort) {
        if (r.timeUs > 0) maxCiPercent = Math.max(maxCiPercent, (r.timeCi / r.timeUs) * 100);
    }

    let csv = "key,value\n";
    csv += `timestamp,${timestamp}\n`;
    csv += `machine,${machine}\n`;
    csv += `n,${N}\n`;
    csv += `max_ci,${maxCiPercent.toFixed(2)}\n`;
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

    // Warmup
    const baseUsers = generateSortedUsers(N);
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
        deltasort: [],
    };

    for (const k of DELTA_COUNTS) {
        process.stdout.write(`  k=${k.toString().padStart(5)}...`);

        const native = runBenchmark(baseUsers, k, (arr, _dirty, cmp) => {
            arr.sort(cmp);
        });
        results.native.push({ k, ...native });

        const bis = runBenchmark(baseUsers, k, binaryInsertionSort);
        results.bis.push({ k, ...bis });

        const esm = runBenchmark(baseUsers, k, extractSortMerge);
        results.esm.push({ k, ...esm });

        const ds = runBenchmark(baseUsers, k, (arr, dirty, cmp) => deltaSort(arr, dirty, cmp));
        results.deltasort.push({ k, ...ds });

        console.log(" done");
    }

    printExecutionTimeTable(results);

    // --- Crossover Analysis (All Algorithms vs Native) ---
    console.log();
    console.log("Running crossover analysis: All algorithms vs Native...");
    const crossoverAllResults: CrossoverResultsAll[] = [];

    for (const size of CROSSOVER_SIZES) {
        process.stdout.write(`  n=${formatNumber(size).padStart(10)}...`);

        const bisKc = findCrossoverBis(size);
        const esmKc = findCrossoverEsm(size);
        const dsKc = findCrossover(size);

        crossoverAllResults.push({
            n: size,
            bisKc,
            bisRatio: (bisKc / size) * 100,
            esmKc,
            esmRatio: (esmKc / size) * 100,
            deltasortKc: dsKc,
            deltasortRatio: (dsKc / size) * 100,
        });
        console.log(
            ` BIS=${((bisKc / size) * 100).toFixed(1)}%, ESM=${((esmKc / size) * 100).toFixed(1)}%, DS=${((dsKc / size) * 100).toFixed(1)}%`,
        );
    }

    printCrossoverTableAll(crossoverAllResults);

    // --- Export CSVs ---
    if (shouldExport) {
        console.log();
        console.log("Exporting CSV files...");
        const basePath = path.join(process.cwd(), "../paper/figures/js");
        fs.mkdirSync(basePath, { recursive: true });
        exportExecutionTimeCsv(results, path.join(basePath, "execution-time.csv"));
        exportCrossoverAllCsv(crossoverAllResults, path.join(basePath, "crossover-all.csv"));
        exportMetadataCsv(results, path.join(basePath, "benchmark_metadata.csv"));
    }

    console.log();
    console.log("Done!");
    if (!shouldExport) {
        console.log("Run with --export to write CSV files to paper/figures/js/");
    }
}

main().catch(console.error);
