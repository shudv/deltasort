import { deltaSort } from "./DeltaSort";

// Test various scales and delta volumes
const SCALE = [1, 2, 3, 4];
const DELTA_VOLUME = [0, 1, 5, 10, 20, 50, 80];
const ITERATION = 10;

describe("DeltaSort", () => {
    describe.each(SCALE)("Scale 10^%d", (scale) => {
        describe.each(DELTA_VOLUME)("Delta %d%", (deltaVolume) => {
            test(`correctness`, () => {
                for (let iter = 0; iter < ITERATION; iter++) {
                    const size = 10 ** scale;
                    // Minimum one delta to ensure some change, otherwise volume proportional to size
                    const deltaCount = Math.max(1, Math.floor((deltaVolume / 100.0) * size));
                    const array = Array.from({ length: size }, (_, i) => i);

                    // Generate distinct dirty indices via Fisher-Yates partial shuffle
                    const pool = Array.from({ length: size }, (_, i) => i);
                    for (let i = 0; i < deltaCount; i++) {
                        const j = i + Math.floor(Math.random() * (size - i));
                        [pool[i], pool[j]] = [pool[j]!, pool[i]!];
                    }
                    const dirtyIndices = pool.slice(0, deltaCount);

                    // Edit dirty values
                    for (const idx of dirtyIndices) {
                        array[idx] = Math.floor(Math.random() * size);
                    }

                    // Create a copy and sort it natively for verification
                    const expected = [...array].sort((a, b) => a - b);

                    // Sort using DeltaSort - ensure that it does not fallaback to native sort by setting high thresholds
                    deltaSort(array, dirtyIndices, (a, b) => a - b);

                    // Verify that the array is correctly sorted
                    for (let i = 1; i < array.length; i++) {
                        expect(array[i]).toBe(expected[i]);
                    }
                }
            });
        });
    });

    test("no dirty indices - should be no-op", () => {
        // deltaSort bails out if no dirty indices are provided
        expect(deltaSort([1, 2, 3, 2, 1], [], (a, b) => a - b)).toEqual([1, 2, 3, 2, 1]);
    });
}, 200000);
