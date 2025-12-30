const SMALL_ARRAY_SORT_THRESHOLD = 256;

enum Direction {
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
    deltaIndices: Set<number>,
): T[] {
    if (
        arr.length <= SMALL_ARRAY_SORT_THRESHOLD ||
        deltaIndices.size > 10000 ||
        deltaIndices.size === 0
    ) {
        return arr.sort(cmp);
    }

    //console.log("===== deltasort start =====");
    //console.log("initial array:", JSON.stringify(arr));
    //console.log("deltaIndices:", [...deltaIndices]);

    const dirty = Array.from(deltaIndices).sort((a, b) => a - b);

    //console.log("sorted dirty indices:", dirty);

    // Sort dirty values & reinsert
    const values = dirty.map((i) => arr[i]!).sort(cmp);
    //console.log("sorted dirty values:", values);

    for (let i = 0; i < dirty.length; i++) {
        //console.log(`reinsert arr[${dirty[i]}] = ${String(values[i])}`);
        arr[dirty[i]!] = values[i]!;
    }

    //console.log("after reinsertion:", JSON.stringify(arr));

    const stack: number[] = [];
    let leftBound = 0;

    function fixLeft(i: number) {
        const v = arr[i]!;
        //console.log(`\n[fixLeft] i=${i}, v=${String(v)}, leftBound=${leftBound}`);
        //console.log("before:", JSON.stringify(arr));

        const target = findLeftTarget(arr, v, leftBound, i - 1, cmp);

        //console.log(`[fixLeft] copyWithin(${target + 1}, ${target}, ${i})`);
        arr.copyWithin(target + 1, target, i);
        arr[target] = v;

        leftBound = target + 1;

        //console.log(`[fixLeft] new leftBound=${leftBound}`);
        //console.log("after:", JSON.stringify(arr));
    }

    function fixRight(i: number, rightBound: number) {
        const v = arr[i]!;
        //console.log(`\n[fixRight] i=${i}, v=${String(v)}, rightBound=${rightBound}`);
        //console.log("before:", JSON.stringify(arr));

        const target = findRightTarget(arr, v, i + 1, rightBound, cmp);

        //console.log(`[fixRight] copyWithin(${i}, ${i + 1}, ${target + 1})`);
        arr.copyWithin(i, i + 1, target + 1);
        arr[target] = v;

        //console.log("after:", JSON.stringify(arr));
    }

    // Main scan
    for (let p = 0; p < dirty.length; p++) {
        const i = dirty[p]!;
        //console.log(`\n[scan] p=${p}, dirtyIndex=${i}`);
        //console.log("[scan] stack:", stack.slice());
        //console.log("[scan] leftBound:", leftBound);
        //console.log("[scan] array:", JSON.stringify(arr));

        const d = directionAt(arr, i, cmp);

        if (d === Direction.LEFT) {
            const rightBound = i - 1;

            //console.log(`[scan] L detected → flushing stack, rightBound=${rightBound}`);

            while (stack.length) {
                const s = stack.pop()!;
                const d = directionAt(arr, s, cmp);

                if (d === Direction.RIGHT) {
                    fixRight(s, rightBound);
                } else {
                    // S → no-op (already in correct position)
                    //console.log(`[flush] skip stable index=${s}`);
                }
            }

            //console.log(`[scan] fixing L index=${i}`);
            fixLeft(i);
        } else {
            //console.log(`[scan] ${d} → push ${i} to stack`);
            stack.push(i);
        }
    }

    // Final flush
    //console.log("\n[final flush]");
    const finalRightBound = arr.length - 1;
    while (stack.length) {
        const s = stack.pop()!;
        //console.log(`[final flush] fixing index=${s}`);
        fixRight(s, finalRightBound);
    }

    //console.log("===== deltasort end =====");
    //console.log("final array:", JSON.stringify(arr));

    return arr;
}

function directionAt<T>(arr: T[], i: number, cmp: (a: T, b: T) => number): Direction {
    const v = arr[i]!;

    const leftBad = i > 0 && cmp(arr[i - 1]!, v) > 0;
    const rightBad = i < arr.length - 1 && cmp(v, arr[i + 1]!) > 0;

    const d = leftBad ? Direction.LEFT : rightBad ? Direction.RIGHT : Direction.STABLE;

    //console.log(`[dir] index=${i}, value=${String(v)}, leftBad=${leftBad}, rightBad=${rightBad} → ${d}`);

    return d;
}

function findLeftTarget<T>(
    arr: T[],
    value: T,
    lo: number,
    hi: number,
    cmp: (a: T, b: T) => number,
): number {
    //console.log(`[findLeftTarget] value=${String(value)}, range=[${lo}, ${hi}]`);

    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const c = cmp(value, arr[mid]!);
        //console.log(`  mid=${mid}, arr[mid]=${String(arr[mid])}, cmp=${c}`);

        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }

    //console.log(`[findLeftTarget] → target=${lo}`);
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
        //console.log(`  mid=${mid}, arr[mid]=${String(arr[mid])}, cmp=${c}`);

        if (c <= 0) lo = mid + 1;
        else hi = mid - 1;
    }

    //console.log(`[findRightTarget] → target=${hi}`);
    return hi;
}
