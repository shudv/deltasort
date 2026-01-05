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
                    const dirtyIndices = new Set<number>();

                    // Edit random deltaCount values
                    for (let i = 0; i < deltaCount; i++) {
                        const indexToEdit = Math.floor(Math.random() * size);
                        array[indexToEdit] = Math.floor(Math.random() * size);
                        dirtyIndices.add(indexToEdit);
                    }

                    // Create a copy and sort it natively for verification
                    const expected = [...array].sort((a, b) => a - b);

                    // Sort using DeltaSort - ensure that it does not fallaback to native sort by setting high thresholds
                    deltaSort(array, (a, b) => a - b, dirtyIndices);

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
        expect(deltaSort([1, 2, 3, 2, 1], (a, b) => a - b, new Set())).toEqual([1, 2, 3, 2, 1]);
    });
}, 200000);
