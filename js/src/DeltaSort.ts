/**
 * Directions for updated indices
 */
const enum Direction {
    /** Value must move left */
    LEFT = 0,
    /** Value may move right and cannot move left */
    RIGHT = 1,
}

/**
 * Sort a previously sorted array using the provided comparator.
 *
 * @param arr The array which was previously sorted and has some updated indices
 * @param updatedIndices Array of indices in the array which have been updated (need not be sorted)
 * @param cmp The comparator function
 *
 * @returns The sorted array
 */
export function deltaSort<T>(arr: T[], updatedIndices: number[], cmp: (a: T, b: T) => number): T[] {
    if (updatedIndices.length === 0) {
        return arr;
    }

    // Sort dirty indices — O(k log k)
    const dirty = updatedIndices.slice().sort((a, b) => a - b);
    const k = dirty.length;

    // Phase 1: Extract dirty values, sort (stable), write back — O(k log k) time, O(k) space
    const values: T[] = new Array(k);
    for (let i = 0; i < k; i++) {
        values[i] = arr[dirty[i]!]!;
    }
    values.sort(cmp);
    for (let i = 0; i < k; i++) {
        arr[dirty[i]!] = values[i]!;
    }

    // Phase 2: Fix ordering violations left to right — O(1) aux
    let leftBound = 0;
    let rightSegStart = -1; // -1 means no active RIGHT segment

    for (let d = 0; d < k; d++) {
        const i = dirty[d]!;
        const direction = getDirection(arr, i, cmp);

        switch (direction) {
            case Direction.LEFT: {
                // Flush pending RIGHT segment in reverse order
                if (rightSegStart >= 0) {
                    let rightBound = i - 1;
                    for (let rd = d - 1; rd >= rightSegStart; rd--) {
                        const idx = dirty[rd]!;
                        if (idx < arr.length - 1 && cmp(arr[idx]!, arr[idx + 1]!) > 0) {
                            rightBound = fixRight(arr, idx, rightBound, cmp) - 1;
                        }
                    }
                    rightSegStart = -1;
                }
                leftBound = fixLeft(arr, i, leftBound, cmp) + 1;
                break;
            }
            case Direction.RIGHT:
                if (rightSegStart < 0) {
                    rightSegStart = d;
                    leftBound = i;
                }
                break;
        }
    }

    // Flush trailing RIGHT segment
    if (rightSegStart >= 0) {
        let rightBound = arr.length - 1;
        for (let rd = k - 1; rd >= rightSegStart; rd--) {
            const idx = dirty[rd]!;
            if (idx < arr.length - 1 && cmp(arr[idx]!, arr[idx + 1]!) > 0) {
                rightBound = fixRight(arr, idx, rightBound, cmp) - 1;
            }
        }
    }

    return arr;
}

/**
 * Determines the direction at updated index i
 *
 * @param arr The array
 * @param i The updated index
 * @param cmp The comparator function
 * @returns The direction
 *
 * @comment This should only be called for an updated index (it does not have any meaning for non-updated indices)
 */
function getDirection<T>(arr: T[], i: number, cmp: (a: T, b: T) => number): Direction {
    return i > 0 && cmp(arr[i - 1]!, arr[i]!) > 0 ? Direction.LEFT : Direction.RIGHT;
}

/**
 * Fixes a RIGHT direction at i by moving it to the correct position
 * between index and rightBound.
 *
 * @param arr The array
 * @param i The index of the RIGHT direction
 * @param rightBound The right boundary for search
 * @param cmp The comparator function
 *
 * @return The new index of the moved value
 */
function fixRight<T>(arr: T[], i: number, rightBound: number, cmp: (a: T, b: T) => number): number {
    const value = arr[i]!;

    // Binary search for target position on the right
    let lo = i + 1;
    let hi = rightBound;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(arr[mid]!, value);
        if (c <= 0) lo = mid + 1;
        else hi = mid - 1;
    }

    move(arr, i, hi);
    return hi;
}

/**
 * Fixes a LEFT direction at i by moving it to the correct position
 * between leftBound and i.
 *
 * @param arr The array
 * @param i The index of the LEFT direction
 * @param leftBound The left boundary for search
 * @param cmp The comparator function
 *
 * @return The new index of the moved value
 */
function fixLeft<T>(arr: T[], i: number, leftBound: number, cmp: (a: T, b: T) => number): number {
    const value = arr[i]!;

    // Binary search for target position on the left
    let lo = leftBound;
    let hi = i - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(value, arr[mid]!);
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }

    move(arr, i, lo);
    return lo;
}

/**
 * Moves value from i `from` to i `to` in the array.
 */
function move<T>(arr: T[], from: number, to: number) {
    // splice-based move implementation (turns out to be faster than copyWithin)
    const [v] = arr.splice(from, 1);
    arr.splice(to, 0, v!);

    // copyWithin-based move implementation
    // if (from < to) {
    //     const v = arr[from]!;
    //     arr.copyWithin(from, from + 1, to + 1);
    //     arr[to] = v;
    // } else if (from > to) {
    //     const v = arr[from]!;
    //     arr.copyWithin(to + 1, to, from);
    //     arr[to] = v;
    // }
}
