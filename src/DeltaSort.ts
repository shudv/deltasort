const enum Direction {
    LEFT = 0,
    RIGHT = 1,
    STABLE = 2,
}

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

    // Step 1: Extract and sort dirty values
    const dirty = Array.from(dirtyIndices).sort((a, b) => a - b);
    const values = dirty.map((i) => arr[i]!).sort(cmp);
    for (let i = 0; i < dirty.length; i++) {
        arr[dirty[i]!] = values[i]!;
    }

    const stack = new Array<number>(dirty.length);
    let stackTop = 0;
    let leftBound = 0;

    // Step 2: Scan dirty indices
    for (let p = 0; p < dirty.length; p++) {
        const i = dirty[p]!;
        const d = directionAt(arr, i, cmp);

        if (d === Direction.LEFT) {
            const rightBound = i - 1;

            // Flush pending RIGHT indices
            while (stackTop > 0) {
                const s = stack[--stackTop]!;
                if (directionAt(arr, s, cmp) === Direction.RIGHT) {
                    const v = arr[s]!;
                    const target = findRightTarget(arr, v, s + 1, rightBound, cmp);
                    move(arr, s, target);
                }
            }

            // Fix LEFT index
            const v = arr[i]!;
            const target = findLeftTarget(arr, v, leftBound, i - 1, cmp);
            move(arr, i, target);
            leftBound = target + 1;
        } else {
            stack[stackTop++] = i;
        }
    }

    // Step 3: Flush remaining RIGHT indices
    const finalRightBound = arr.length - 1;
    while (stackTop > 0) {
        const s = stack[--stackTop]!;
        if (directionAt(arr, s, cmp) === Direction.RIGHT) {
            const v = arr[s]!;
            const target = findRightTarget(arr, v, s + 1, finalRightBound, cmp);
            move(arr, s, target);
        }
    }

    return arr;
}

function directionAt<T>(arr: T[], i: number, cmp: (a: T, b: T) => number): Direction {
    return i > 0 && cmp(arr[i - 1]!, arr[i]!) > 0
        ? Direction.LEFT
        : i < arr.length - 1 && cmp(arr[i]!, arr[i + 1]!) > 0
          ? Direction.RIGHT
          : Direction.STABLE;
}

function findLeftTarget<T>(
    arr: T[],
    value: T,
    lo: number,
    hi: number,
    cmp: (a: T, b: T) => number,
): number {
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(value, arr[mid]!);
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }

    return lo;
}

function findRightTarget<T>(
    arr: T[],
    value: T,
    lo: number,
    hi: number,
    cmp: (a: T, b: T) => number,
): number {
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(arr[mid]!, value);
        if (c <= 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return hi;
}

function move<T>(arr: T[], from: number, to: number) {
    const [v] = arr.splice(from, 1);
    arr.splice(to, 0, v!);
}
