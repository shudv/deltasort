import { User, generateSortedUsers, mutateUser, userComparator } from "./BenchmarkData";
import { deltasort } from "./DeltaSort";

// ============================================================================
// BENCHMARK CONFIGURATION
// ============================================================================

/** Array size to test */
const N = 50000;

/** Delta counts to test */
const DELTA_COUNTS = [1, 2, 5, 10, 20, 50, 100];

/** Number of iterations per test configuration */
const ITERATIONS = 100;

// ============================================================================
// SORTING ALGORITHMS
// ============================================================================

type Algorithm = "native" | "binaryInsertion" | "extractSortMerge" | "deltasort";

/**
 * Binary Insertion: For each dirty index, remove the element,
 * find its correct position via binary search, and insert it.
 * Process one element at a time to handle index shifts correctly.
 */
function binaryInsertionSort<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
): T[] {
    // Correct approach: Extract all dirty elements first (descending order to preserve indices)
    const sortedDesc = Array.from(dirtyIndices).sort((a, b) => b - a);
    const elements: T[] = [];

    for (const idx of sortedDesc) {
        elements.push(arr[idx]!);
        arr.splice(idx, 1);
    }

    // Now insert each element at its correct position
    // Elements were extracted in descending index order, so reverse to process in original order
    // Actually, order doesn't matter for correctness since we binary search each time
    for (const value of elements) {
        const insertIdx = binarySearchPosition(arr, value, cmp);
        arr.splice(insertIdx, 0, value);
    }

    return arr;
}

/**
 * Extract-Sort-Merge: Extract dirty elements, sort them separately,
 * then merge with the clean portion of the array.
 */
function extractSortMerge<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
): T[] {
    // Extract dirty elements (descending order to preserve indices during removal)
    const sortedIndices = Array.from(dirtyIndices).sort((a, b) => b - a);
    const dirtyElements: T[] = [];

    for (const idx of sortedIndices) {
        dirtyElements.push(arr[idx]!);
        arr.splice(idx, 1);
    }

    // Sort the dirty elements
    dirtyElements.sort(cmp);

    // Merge: arr is already sorted (clean elements), dirtyElements is sorted
    const result: T[] = [];
    let i = 0;
    let j = 0;

    while (i < arr.length && j < dirtyElements.length) {
        if (cmp(arr[i]!, dirtyElements[j]!) <= 0) {
            result.push(arr[i]!);
            i++;
        } else {
            result.push(dirtyElements[j]!);
            j++;
        }
    }

    while (i < arr.length) {
        result.push(arr[i]!);
        i++;
    }

    while (j < dirtyElements.length) {
        result.push(dirtyElements[j]!);
        j++;
    }

    // Copy back to original array
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

interface AlgorithmResult {
    time: number;
    comparisons: number;
}

interface AlgorithmStats {
    timeMean: number;
    timeVariance: number;
    comparisons: number;
}

interface BenchmarkResult {
    deltaCount: number;
    native: AlgorithmStats;
    binaryInsertion: AlgorithmStats;
    extractSortMerge: AlgorithmStats;
    deltasort: AlgorithmStats;
}

function calculateStats(values: number[]): { mean: number; variancePercent: number } {
    const n = values.length;
    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n;
    const stdDev = Math.sqrt(variance);
    const variancePercent = mean > 0 ? (stdDev / mean) * 100 : 0;
    return { mean, variancePercent };
}

function runAlgorithm<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
    algorithm: Algorithm,
): AlgorithmResult {
    let comparisons = 0;
    const countingCmp = (a: T, b: T): number => {
        comparisons++;
        return cmp(a, b);
    };

    const start = performance.now();

    switch (algorithm) {
        case "native":
            arr.sort(countingCmp); // Unlike other sorts, native sort is blind to dirty indices
            break;
        case "binaryInsertion":
            binaryInsertionSort(arr, countingCmp, dirtyIndices);
            break;
        case "extractSortMerge":
            extractSortMerge(arr, countingCmp, dirtyIndices);
            break;
        case "deltasort":
            deltasort(arr, countingCmp, dirtyIndices);
            break;
    }

    const end = performance.now();

    // Verify that the array is correctly sorted
    for (let i = 1; i < arr.length; i++) {
        if (cmp(arr[i - 1]!, arr[i]!) > 0) {
            throw new Error(`Array is not sorted correctly by ${algorithm} algorithm.`);
        }
    }
    return { time: end - start, comparisons };
}

// ============================================================================
// REPORTING
// ============================================================================

function printHeader(title: string): void {
    console.log("\n" + "═".repeat(120));
    console.log(`  ${title}`);
    console.log("═".repeat(120));
}

function formatTimeWithVariance(mean: number, variancePercent: number): string {
    return `${mean.toFixed(3)} ± ${variancePercent.toFixed(0)}%`;
}

function formatTimeTable(results: BenchmarkResult[]): string {
    const headers = ["#Changes", "NativeSort", "BinaryInsertion", "ExtractSortMerge", "DeltaSort"];
    const rows = results.map((r) => [
        r.deltaCount.toString(),
        formatTimeWithVariance(r.native.timeMean, r.native.timeVariance),
        formatTimeWithVariance(r.binaryInsertion.timeMean, r.binaryInsertion.timeVariance),
        formatTimeWithVariance(r.extractSortMerge.timeMean, r.extractSortMerge.timeVariance),
        formatTimeWithVariance(r.deltasort.timeMean, r.deltasort.timeVariance),
    ]);

    return formatTableRows(headers, rows);
}

function formatComparisonTable(results: BenchmarkResult[]): string {
    const headers = ["#Changes", "NativeSort", "BinaryInsertion", "ExtractSortMerge", "DeltaSort"];
    const rows = results.map((r) => [
        r.deltaCount.toString(),
        r.native.comparisons.toLocaleString(),
        r.binaryInsertion.comparisons.toLocaleString(),
        r.extractSortMerge.comparisons.toLocaleString(),
        r.deltasort.comparisons.toLocaleString(),
    ]);

    return formatTableRows(headers, rows);
}

function formatSpeedupTable(results: BenchmarkResult[]): string {
    const headers = ["#Changes", "BinaryInsertion", "ExtractSortMerge", "DeltaSort", "Best"];
    const rows = results.map((r) => {
        const binIns = r.native.timeMean / r.binaryInsertion.timeMean;
        const extSort = r.native.timeMean / r.extractSortMerge.timeMean;
        const delta = r.native.timeMean / r.deltasort.timeMean;

        const times = [
            { name: "NativeSort", time: r.native.timeMean },
            { name: "BinaryInsertion", time: r.binaryInsertion.timeMean },
            { name: "ExtractSortMerge", time: r.extractSortMerge.timeMean },
            { name: "DeltaSort", time: r.deltasort.timeMean },
        ];
        const best = times.reduce((a, b) => (a.time < b.time ? a : b));

        return [
            r.deltaCount.toString(),
            `${binIns.toFixed(2)}x`,
            `${extSort.toFixed(2)}x`,
            `${delta.toFixed(2)}x`,
            best.name,
        ];
    });

    return formatTableRows(headers, rows);
}

function formatTableRows(headers: string[], rows: string[][]): string {
    const colWidths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
    const separator = "+" + colWidths.map((w) => "-".repeat(w + 2)).join("+") + "+";
    const headerRow = "|" + headers.map((h, i) => ` ${h.padEnd(colWidths[i]!)} `).join("|") + "|";
    const dataRows = rows.map(
        (row) => "|" + row.map((cell, i) => ` ${cell.padStart(colWidths[i]!)} `).join("|") + "|",
    );

    return [separator, headerRow, separator, ...dataRows, separator].join("\n");
}

// ============================================================================
// BENCHMARK TEST
// ============================================================================

describe("DeltaSort Benchmark", () => {
    test("Performance Comparison: NativeSort vs BinaryInsertion vs ExtactSortMerge vs DeltaSort", () => {
        printHeader("DELTASORT BENCHMARK SUITE");

        console.log("\n  Configuration:");
        console.log(`    • Array Size (N):        ${N.toLocaleString()}`);
        console.log(`    • Delta Counts:          [${DELTA_COUNTS.join(", ")}]`);
        console.log(`    • Iterations per test:   ${ITERATIONS}`);
        console.log(`    • Data Type:             User objects (country, age, name)`);

        const results: BenchmarkResult[] = [];
        const baseUsers = generateSortedUsers(N);

        for (const deltaCount of DELTA_COUNTS) {
            const nativeResults: AlgorithmResult[] = [];
            const binInsResults: AlgorithmResult[] = [];
            const extSortResults: AlgorithmResult[] = [];
            const deltasortResults: AlgorithmResult[] = [];

            for (let iter = 0; iter < ITERATIONS; iter++) {
                const usersForNativeSort = baseUsers.map((u) => ({ ...u }));
                const usersForBinIns = baseUsers.map((u) => ({ ...u }));
                const usersForExtSort = baseUsers.map((u) => ({ ...u }));
                const usersForDeltasort = baseUsers.map((u) => ({ ...u }));

                const dirtyIndices = new Set<number>();
                const mutations: { idx: number; newUser: User }[] = [];

                for (let i = 0; i < deltaCount; i++) {
                    const idx = Math.floor(Math.random() * N);
                    const newUser = mutateUser(usersForNativeSort[idx]!);
                    mutations.push({ idx, newUser });
                    dirtyIndices.add(idx);
                }

                for (const { idx, newUser } of mutations) {
                    usersForNativeSort[idx] = { ...newUser };
                    usersForBinIns[idx] = { ...newUser };
                    usersForExtSort[idx] = { ...newUser };
                    usersForDeltasort[idx] = { ...newUser };
                }

                nativeResults.push(
                    runAlgorithm(usersForNativeSort, userComparator, dirtyIndices, "native"),
                );
                binInsResults.push(
                    runAlgorithm(usersForBinIns, userComparator, dirtyIndices, "binaryInsertion"),
                );
                extSortResults.push(
                    runAlgorithm(usersForExtSort, userComparator, dirtyIndices, "extractSortMerge"),
                );
                deltasortResults.push(
                    runAlgorithm(usersForDeltasort, userComparator, dirtyIndices, "deltasort"),
                );

                if (iter === 0) {
                    for (let i = 0; i < N; i++) {
                        expect(usersForBinIns[i]).toEqual(usersForNativeSort[i]);
                        expect(usersForExtSort[i]).toEqual(usersForNativeSort[i]);
                        expect(usersForDeltasort[i]).toEqual(usersForNativeSort[i]);
                    }
                }
            }

            const aggregate = (res: AlgorithmResult[]): AlgorithmStats => {
                const stats = calculateStats(res.map((r) => r.time));
                return {
                    timeMean: stats.mean,
                    timeVariance: stats.variancePercent,
                    comparisons: Math.round(
                        res.reduce((sum, r) => sum + r.comparisons, 0) / ITERATIONS,
                    ),
                };
            };

            results.push({
                deltaCount,
                native: aggregate(nativeResults),
                binaryInsertion: aggregate(binInsResults),
                extractSortMerge: aggregate(extSortResults),
                deltasort: aggregate(deltasortResults),
            });
        }

        // Print Results
        printHeader("RESULTS: Execution Time (ms)");
        console.log("\n" + formatTimeTable(results));

        printHeader("RESULTS: Comparator Calls");
        console.log("\n" + formatComparisonTable(results));

        printHeader("RESULTS: Speedup vs NativeSort (higher is better)");
        console.log("\n" + formatSpeedupTable(results));

        console.log("\n" + "═".repeat(120) + "\n");
    });
}, 600000);
