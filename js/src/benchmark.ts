import { User, generateSortedUsers, mutateUser, userComparator } from "./BenchmarkData";
import { deltaSort } from "./DeltaSort";
import * as fs from "fs";
import * as path from "path";

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

const N = 50_000;
const DELTA_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500];
const CROSSOVER_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000];

// For managed environments we need higher iterations to reduce variance
const ITERATIONS = 500;
const CROSSOVER_ITERATIONS = 50;

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
// BENCHMARK INFRASTRUCTURE
// ============================================================================

interface ExecutionTimeResult {
    k: number;
    native: number;
    bis: number;
    esm: number;
    deltasort: number;
}

interface ComparatorCountResult {
    k: number;
    native: number;
    bis: number;
    esm: number;
    deltasort: number;
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

function measureTime(fn: () => void): number {
    const start = performance.now();
    fn();
    return (performance.now() - start) * 1000; // Convert to µs
}

// ============================================================================
// CROSSOVER ANALYSIS
// ============================================================================

function deltasortIsFaster(baseUsers: User[], k: number, n: number): boolean {
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

    if (!deltasortIsFaster(baseUsers, 1, n)) {
        return 0;
    }

    if (deltasortIsFaster(baseUsers, n, n)) {
        return n;
    }

    const minRange = Math.floor(n * 0.0001);

    while (lo < hi) {
        if (hi - lo < minRange) {
            break;
        }

        const mid = lo + Math.floor((hi - lo + 1) / 2);

        if (deltasortIsFaster(baseUsers, mid, n)) {
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

function printExecutionTimeTable(results: ExecutionTimeResult[]): void {
    console.log();
    console.log("Execution Time (µs) - n=50,000");
    console.log("┌────────┬──────────┬──────────┬──────────┬──────────┐");
    console.log("│   k    │  Native  │   BIS    │   ESM    │ DeltaSort│");
    console.log("├────────┼──────────┼──────────┼──────────┼──────────┤");
    for (const r of results) {
        console.log(
            `│ ${r.k.toString().padStart(6)} │ ${r.native.toFixed(1).padStart(8)} │ ${r.bis.toFixed(1).padStart(8)} │ ${r.esm.toFixed(1).padStart(8)} │ ${r.deltasort.toFixed(1).padStart(8)} │`,
        );
    }
    console.log("└────────┴──────────┴──────────┴──────────┴──────────┘");
}

function printComparatorCountTable(results: ComparatorCountResult[]): void {
    console.log();
    console.log("Comparator Invocations - n=50,000");
    console.log("┌────────┬──────────┬──────────┬──────────┬──────────┐");
    console.log("│   k    │  Native  │   BIS    │   ESM    │ DeltaSort│");
    console.log("├────────┼──────────┼──────────┼──────────┼──────────┤");
    for (const r of results) {
        console.log(
            `│ ${r.k.toString().padStart(6)} │ ${r.native.toString().padStart(8)} │ ${r.bis.toString().padStart(8)} │ ${r.esm.toString().padStart(8)} │ ${r.deltasort.toString().padStart(8)} │`,
        );
    }
    console.log("└────────┴──────────┴──────────┴──────────┴──────────┘");
}

function printCrossoverTable(results: CrossoverResult[]): void {
    console.log();
    console.log("Crossover Threshold Analysis");
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
// CSV EXPORT
// ============================================================================

function exportExecutionTimeCsv(results: ExecutionTimeResult[], filePath: string): void {
    let csv = "k,native,bis,esm,deltasort\n";
    for (const r of results) {
        csv += `${r.k},${r.native.toFixed(1)},${r.bis.toFixed(1)},${r.esm.toFixed(1)},${r.deltasort.toFixed(1)}\n`;
    }
    fs.writeFileSync(filePath, csv);
    console.log(`Exported: ${filePath}`);
}

function exportComparatorCountCsv(results: ComparatorCountResult[], filePath: string): void {
    let csv = "k,native,bis,esm,deltasort\n";
    for (const r of results) {
        csv += `${r.k},${r.native},${r.bis},${r.esm},${r.deltasort}\n`;
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

    // --- Execution Time ---
    console.log();
    console.log("Running execution time benchmarks...");
    const execResults: ExecutionTimeResult[] = [];

    for (const k of DELTA_COUNTS) {
        process.stdout.write(`  k=${k.toString().padStart(5)}...`);

        const indices = sampleDistinctIndices(N, k);
        const mutatedUsers = baseUsers.map((u) => ({ ...u }));
        const dirtyIndices = new Set<number>();

        for (const idx of indices) {
            mutatedUsers[idx] = mutateUser(mutatedUsers[idx]!);
            dirtyIndices.add(idx);
        }

        let nativeTotal = 0,
            bisTotal = 0,
            esmTotal = 0,
            dsTotal = 0;

        for (let i = 0; i < ITERATIONS; i++) {
            const usersNative = mutatedUsers.map((u) => ({ ...u }));
            const usersBis = mutatedUsers.map((u) => ({ ...u }));
            const usersEsm = mutatedUsers.map((u) => ({ ...u }));
            const usersDs = mutatedUsers.map((u) => ({ ...u }));
            const dirty = new Set(dirtyIndices);

            nativeTotal += measureTime(() => usersNative.sort(userComparator));
            bisTotal += measureTime(() => binaryInsertionSort(usersBis, userComparator, dirty));
            esmTotal += measureTime(() =>
                extractSortMerge(usersEsm, userComparator, new Set(dirtyIndices)),
            );
            dsTotal += measureTime(() => deltaSort(usersDs, dirtyIndices, userComparator));
        }

        execResults.push({
            k,
            native: nativeTotal / ITERATIONS,
            bis: bisTotal / ITERATIONS,
            esm: esmTotal / ITERATIONS,
            deltasort: dsTotal / ITERATIONS,
        });
        console.log(" done");
    }

    printExecutionTimeTable(execResults);

    // --- Comparator Count ---
    console.log();
    console.log("Running comparator count benchmarks...");
    const cmpResults: ComparatorCountResult[] = [];

    for (const k of DELTA_COUNTS) {
        process.stdout.write(`  k=${k.toString().padStart(5)}...`);

        const indices = sampleDistinctIndices(N, k);
        const mutatedUsers = baseUsers.map((u) => ({ ...u }));
        const dirtyIndices = new Set<number>();

        for (const idx of indices) {
            mutatedUsers[idx] = mutateUser(mutatedUsers[idx]!);
            dirtyIndices.add(idx);
        }

        const usersNative = mutatedUsers.map((u) => ({ ...u }));
        const usersBis = mutatedUsers.map((u) => ({ ...u }));
        const usersEsm = mutatedUsers.map((u) => ({ ...u }));
        const usersDs = mutatedUsers.map((u) => ({ ...u }));

        let nativeCmp = 0,
            bisCmp = 0,
            esmCmp = 0,
            dsCmp = 0;

        const countingCmp =
            (counter: { count: number }) =>
            (a: User, b: User): number => {
                counter.count++;
                return userComparator(a, b);
            };

        const nativeCounter = { count: 0 };
        usersNative.sort(countingCmp(nativeCounter));
        nativeCmp = nativeCounter.count;

        const bisCounter = { count: 0 };
        binaryInsertionSort(usersBis, countingCmp(bisCounter), new Set(dirtyIndices));
        bisCmp = bisCounter.count;

        const esmCounter = { count: 0 };
        extractSortMerge(usersEsm, countingCmp(esmCounter), new Set(dirtyIndices));
        esmCmp = esmCounter.count;

        const dsCounter = { count: 0 };
        deltaSort(usersDs, dirtyIndices, countingCmp(dsCounter));
        dsCmp = dsCounter.count;

        cmpResults.push({
            k,
            native: nativeCmp,
            bis: bisCmp,
            esm: esmCmp,
            deltasort: dsCmp,
        });
        console.log(" done");
    }

    printComparatorCountTable(cmpResults);

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
        const basePath = path.join(__dirname, "../../paper/benchmarks/js");
        fs.mkdirSync(basePath, { recursive: true });
        exportExecutionTimeCsv(execResults, path.join(basePath, "execution-time.csv"));
        exportComparatorCountCsv(cmpResults, path.join(basePath, "comparator-count.csv"));
        exportCrossoverCsv(crossoverResults, path.join(basePath, "crossover-threshold.csv"));
    }

    console.log();
    console.log("Done!");
    if (!shouldExport) {
        console.log("Run with --export to write CSV files to paper/benchmarks/js/");
    }
}

main().catch(console.error);
