import { deltasort } from "./DeltaSort";

// Configuration for benchmark iterations
const BENCHMARK_ITERATIONS = 500;

interface BenchmarkStats {
    scale: number;
    size: number;
    deltaVolume: number;
    deltaCount: number;
    nativeTimeMean: number;
    nativeTimeStdDev: number;
    deltasortTimeMean: number;
    deltasortTimeStdDev: number;
    speedup: number;
    nativeComparisons: number;
    deltasortComparisons: number;
    comparisonReduction: number;
}

interface IterationResult {
    nativeTime: number;
    deltasortTime: number;
    nativeComparisons: number;
    deltasortComparisons: number;
}

function calculateStats(values: number[]): { mean: number; stdDev: number } {
    const n = values.length;
    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n;
    const stdDev = Math.sqrt(variance);
    return { mean, stdDev };
}

function formatTime(mean: number, stdDev: number): string {
    return `${mean.toFixed(3)} ± ${stdDev.toFixed(3)}`;
}

function formatTable(stats: BenchmarkStats[]): string {
    const headers = [
        "Size",
        "Delta Count",
        "Native (ms)",
        "DeltaSort (ms)",
        "Speedup",
        "Native Cmp",
        "DeltaSort Cmp",
        "Cmp Reduction",
    ];
    const rows = stats.map((s) => [
        s.size.toLocaleString(),
        `${s.deltaVolume}`,
        formatTime(s.nativeTimeMean, s.nativeTimeStdDev),
        formatTime(s.deltasortTimeMean, s.deltasortTimeStdDev),
        `${s.speedup.toFixed(2)}x`,
        s.nativeComparisons.toLocaleString(),
        s.deltasortComparisons.toLocaleString(),
        `${s.comparisonReduction.toFixed(1)}%`,
    ]);

    // Calculate column widths
    const colWidths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));

    // Build table
    const separator = "+" + colWidths.map((w) => "-".repeat(w + 2)).join("+") + "+";
    const headerRow = "|" + headers.map((h, i) => ` ${h.padEnd(colWidths[i]!)} `).join("|") + "|";
    const dataRows = rows.map(
        (row) => "|" + row.map((cell, i) => ` ${cell.padStart(colWidths[i]!)} `).join("|") + "|",
    );

    return [separator, headerRow, separator, ...dataRows, separator].join("\n");
}

const SCALE = [1000];
const DELTA_VOLUME = [0, 1, 5, 10, 20, 50, 80, 100];

describe("DeltaSort", () => {
    describe.each(SCALE)("Scale 10^%d", (scale) => {
        describe.each(DELTA_VOLUME)("Delta %d%", (deltaVolume) => {
            test(`correctness`, () => {
                //console.log(
                //    `\n--- Testing Scale 10^${scale} with Delta Volume ${deltaVolume}% ---`
                //);
                const size = scale;
                // Minimum one delta to ensure some change, otherwise volume proportional to size
                const deltaCount = Math.max(1, Math.floor((deltaVolume / 100) * size));
                const array = Array.from({ length: size }, (_, i) => i);

                //console.log(JSON.stringify(array));

                const dirtyIndices = new Set<number>();

                // Edit random deltaCount elements
                for (let i = 0; i < deltaCount; i++) {
                    const indexToEdit = Math.floor(Math.random() * size);
                    array[indexToEdit] = Math.floor(Math.random() * size);

                    //console.log(
                    //    `Edited index ${indexToEdit}, new value: ${array[indexToEdit]}`
                    //);
                    dirtyIndices.add(indexToEdit);
                }
                deltasort(array, (a, b) => a - b, dirtyIndices);

                // Verify that the array is sorted
                for (let i = 1; i < array.length; i++) {
                    expect(array[i]! >= array[i - 1]!).toBe(true);
                }
            });
        });
    });

    test(`performance`, () => {
        const scales = [4];
        const deltas = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 8000];
        const iterations = BENCHMARK_ITERATIONS;
        const stats: BenchmarkStats[] = [];

        for (const scale of scales) {
            for (const deltaVolume of deltas) {
                const size = 10 ** scale;
                const deltaCount = deltaVolume;
                const iterationResults: IterationResult[] = [];

                for (let iter = 0; iter < iterations; iter++) {
                    // Create fresh array for each iteration
                    const array = Array.from({ length: size }, (_, i) => i);

                    // Edit random deltaCount elements
                    const dirtyIndices: number[] = [];
                    for (let i = 0; i < deltaCount; i++) {
                        const indexToEdit = Math.floor(Math.random() * size);
                        array[indexToEdit] = Math.floor(Math.random() * size);
                        dirtyIndices.push(indexToEdit);
                    }

                    // Create a copy of the array so that we can sort it separately
                    const arrayToSort = array.slice();

                    // Sort using native sort and measure time + comparisons
                    let nativeComparisons = 0;
                    const nativeStart = performance.now();
                    arrayToSort.sort((a, b) => {
                        nativeComparisons++;
                        for (let k = 0; k < 25; k++) {} // Artificial delay
                        return a - b;
                    });
                    const nativeEnd = performance.now();
                    const nativeTime = nativeEnd - nativeStart;

                    // Sort using deltasort and measure time + comparisons
                    const indexSet = new Set(dirtyIndices);
                    let deltasortComparisons = 0;
                    const start = performance.now();
                    const sortedArray = deltasort(
                        array,
                        (a, b) => {
                            deltasortComparisons++;
                            for (let k = 0; k < 25; k++) {} // Artificial delay
                            return a - b;
                        },
                        indexSet,
                    );
                    const end = performance.now();
                    const deltasortTime = end - start;

                    iterationResults.push({
                        nativeTime,
                        deltasortTime,
                        nativeComparisons,
                        deltasortComparisons,
                    });

                    // Verify correctness on first iteration only
                    if (iter === 0) {
                        for (let i = 0; i < sortedArray.length; i++) {
                            expect(sortedArray[i]).toEqual(arrayToSort[i]);
                        }
                    }
                }

                // Calculate aggregate stats across iterations
                const nativeTimeStats = calculateStats(iterationResults.map((r) => r.nativeTime));
                const deltasortTimeStats = calculateStats(
                    iterationResults.map((r) => r.deltasortTime),
                );
                const avgNativeComparisons = Math.round(
                    iterationResults.reduce((sum, r) => sum + r.nativeComparisons, 0) / iterations,
                );
                const avgDeltasortComparisons = Math.round(
                    iterationResults.reduce((sum, r) => sum + r.deltasortComparisons, 0) /
                        iterations,
                );
                const comparisonReduction =
                    avgNativeComparisons > 0
                        ? ((avgNativeComparisons - avgDeltasortComparisons) /
                              avgNativeComparisons) *
                          100
                        : 0;

                stats.push({
                    scale,
                    size,
                    deltaVolume,
                    deltaCount,
                    nativeTimeMean: nativeTimeStats.mean,
                    nativeTimeStdDev: nativeTimeStats.stdDev,
                    deltasortTimeMean: deltasortTimeStats.mean,
                    deltasortTimeStdDev: deltasortTimeStats.stdDev,
                    speedup: nativeTimeStats.mean / deltasortTimeStats.mean,
                    nativeComparisons: avgNativeComparisons,
                    deltasortComparisons: avgDeltasortComparisons,
                    comparisonReduction,
                });
            }
        }

        // Pretty print results as a table
        console.log(
            `\n📊 DeltaSort Performance Benchmark Results (${iterations} iterations per config)\n`,
        );
        console.log(formatTable(stats));

        // Summary statistics
        const avgSpeedup = stats.reduce((sum, s) => sum + s.speedup, 0) / stats.length;
        const maxSpeedup = Math.max(...stats.map((s) => s.speedup));
        const minSpeedup = Math.min(...stats.map((s) => s.speedup));
        const bestCase = stats.find((s) => s.speedup === maxSpeedup)!;
        const worstCase = stats.find((s) => s.speedup === minSpeedup)!;

        console.log("\n📈 Summary Statistics:");
        console.log(`   Average Speedup: ${avgSpeedup.toFixed(2)}x`);
        console.log(
            `   Best Case: ${maxSpeedup.toFixed(2)}x (Size: ${bestCase.size.toLocaleString()}, Delta: ${bestCase.deltaVolume})`,
        );
        console.log(
            `   Worst Case: ${minSpeedup.toFixed(2)}x (Size: ${worstCase.size.toLocaleString()}, Delta: ${worstCase.deltaVolume})`,
        );

        const avgCompReduction =
            stats.reduce((sum, s) => sum + s.comparisonReduction, 0) / stats.length;
        const maxCompReduction = Math.max(...stats.map((s) => s.comparisonReduction));
        const bestCompCase = stats.find((s) => s.comparisonReduction === maxCompReduction)!;
        console.log(`\n   Average Comparison Reduction: ${avgCompReduction.toFixed(1)}%`);
        console.log(
            `   Best Comparison Reduction: ${maxCompReduction.toFixed(1)}% (Size: ${bestCompCase.size.toLocaleString()}, Delta: ${bestCompCase.deltaVolume})`,
        );
        console.log("");
    });
}, 120000);
