/**
 * Sorts the array using the provided comparator, only re-sorting if there are dirty indices.
 * @param sortedArray
 * @param comparator
 * @param dirtyIndices
 * @returns
 */
export function deltasort<T>(
    arr: T[],
    cmp: (a: T, b: T) => number,
    dirtyIndices: Set<number>,
): T[] {
    if (dirtyIndices.size === 0) {
        return arr;
    }

    // Phase 1: Extract and sort dirty values
    const dirty = Array.from(dirtyIndices).sort((a, b) => a - b);
    const values = dirty.map((i) => arr[i]!).sort(cmp);
    for (let i = 0; i < dirty.length; i++) {
        arr[dirty[i]!] = values[i]!;
    }

    // Stack for pending violations
    const pending = new Array<number>(dirty.length);
    let stackTop = 0;

    // Left boundary for fixing LEFT violations, everything up to leftBound is already fixed
    let leftBound = 0;

    // Phase 2: Scan dirty indices left to right
    for (let p = 0; p < dirty.length; p++) {
        const i = dirty[p]!;
        if (isLeftViolation(arr, i, cmp)) {
            // Fix all pending indices before fixing LEFT
            while (stackTop > 0) {
                fixPendingViolation(arr, pending[--stackTop]!, i - 1, cmp);
            }

            // Fix LEFT index
            leftBound = fixLeftViolation(arr, i, leftBound, cmp) + 1;
        } else {
            pending[stackTop++] = i;
        }
    }

    // Fix any remaining RIGHT violations
    while (stackTop > 0) {
        fixPendingViolation(arr, pending[--stackTop]!, arr.length - 1, cmp);
    }

    return arr;
}

function fixPendingViolation<T>(
    arr: T[],
    index: number,
    rightBound: number,
    cmp: (a: T, b: T) => number,
): void {
    if (!isRightViolation(arr, index, cmp)) {
        return;
    }

    const value = arr[index]!;

    // Binary search for target position on the right
    let lo = index + 1;
    let hi = rightBound;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(arr[mid]!, value);
        if (c <= 0) lo = mid + 1;
        else hi = mid - 1;
    }

    move(arr, index, hi);
}

function fixLeftViolation<T>(
    arr: T[],
    index: number,
    leftBound: number,
    cmp: (a: T, b: T) => number,
): number {
    const value = arr[index]!;

    // Binary search for target position on the left
    let lo = leftBound;
    let hi = index - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(value, arr[mid]!);
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }

    move(arr, index, lo);
    return lo;
}

function isLeftViolation<T>(arr: T[], i: number, cmp: (a: T, b: T) => number): boolean {
    return i > 0 && cmp(arr[i - 1]!, arr[i]!) > 0;
}

function isRightViolation<T>(arr: T[], i: number, cmp: (a: T, b: T) => number): boolean {
    return i < arr.length - 1 && cmp(arr[i]!, arr[i + 1]!) > 0;
}

/**
 * Moves element from index `from` to index `to` in the array.
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
