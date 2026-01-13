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
 * @param updatedIndices Set of indices in the array which have been updated
 * @param cmp The comparator function
 *
 * @returns The sorted array
 */
export function deltaSort<T>(
    arr: T[],
    updatedIndices: Set<number>,
    cmp: (a: T, b: T) => number,
): T[] {
    if (updatedIndices.size === 0) {
        return arr;
    }

    // Phase 1: Extract and sort updated values
    const updated = Array.from(updatedIndices).sort((a, b) => a - b);
    const values = updated.map((i) => arr[i]!).sort(cmp);
    for (let i = 0; i < updated.length; i++) {
        arr[updated[i]!] = values[i]!;
    }

    // Add a sentinel at the end to trigger final flush
    updated.push(arr.length);

    // Phase 2: Scan updated indices left to right

    // Stack for pending RIGHT directions
    const pendingRightDirections = new Array<number>(updated.length);
    let stackTop = 0;

    // Left boundary for fixing LEFT violations, everything before the leftBound is already fixed
    let leftBound = 0;

    for (let p = 0; p < updated.length; p++) {
        const i = updated[p]!;

        // Determine direction (sentinel is considered LEFT to trigger final flush)
        const direction = i == arr.length ? Direction.LEFT : getDirection(arr, i, cmp);

        switch (direction) {
            case Direction.LEFT: {
                // Fix all pending indices before fixing LEFT
                let rightBound = i - 1;
                while (stackTop > 0) {
                    const ri = pendingRightDirections[--stackTop]!;
                    // Fix RIGHT direction at ri if needed
                    if (ri < arr.length - 1 && cmp(arr[ri]!, arr[ri + 1]!) > 0) {
                        rightBound = fixRight(arr, ri, rightBound, cmp) - 1;
                    }
                }

                // Fix actual (non-sentinel) LEFT directions
                if (i < arr.length) {
                    leftBound = fixLeft(arr, i, leftBound, cmp) + 1;
                }
                break;
            }
            case Direction.RIGHT:
                if (stackTop === 0) {
                    // First RIGHT in segment advances left bound
                    leftBound = i;
                }
                pendingRightDirections[stackTop++] = i;
                break;
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
