#!/usr/bin/env node

import esbuild from "esbuild";
import { gzipSync } from "zlib";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function formatBytes(b: number) {
    if (b < 1024) return `${b} B`;
    return `${(b / 1024).toFixed(2)} kB`;
}

async function measure(pkg: string, exportName: string | null) {
    const importLine = exportName
        ? `import { ${exportName} } from "${pkg}";\nconsole.log(typeof ${exportName});`
        : `import * as X from "${pkg}";\nconsole.log(typeof X);`;

    const tmpName = `.measure-entry-${Date.now()}.mjs`;
    const tmpPath = path.join(process.cwd(), tmpName);
    fs.writeFileSync(tmpPath, importLine, "utf8");

    try {
        const result = await esbuild.build({
            entryPoints: [tmpPath],
            bundle: true,
            write: false,
            minify: true,
            format: "esm",
            platform: "browser",
            sourcemap: false,
            external: [],
        });

        const out = result.outputFiles?.[0];
        if (!out) throw new Error("esbuild returned no output files");

        const codeBuf = Buffer.from(out.text ?? out.contents);

        return {
            minified: codeBuf.length,
            gz: gzipSync(new Uint8Array(codeBuf)).length,
        };
    } finally {
        fs.unlinkSync(tmpPath);
    }
}

async function main() {
    const args = process.argv.slice(2);
    const [pkg, exportName] = args;

    if (!pkg) {
        console.log("Usage: node measure-size.js <package> [exportName]");
        process.exit(1);
    }

    console.log(`📦 Package: ${pkg}`);
    if (exportName) console.log(`🔹 Export: ${exportName}`);
    console.log("");

    console.log("⏳ Measuring (treeshaken import)...");
    const tree = await measure(pkg, exportName);
    console.log("=== Treeshaken ===");
    console.log("Minified:", formatBytes(tree.minified));
    console.log("Gzipped:", formatBytes(tree.gz));

    console.log("\n⏳ Measuring (full import *)...");
    const full = await measure(pkg, null);
    console.log("=== Full Bundle ===");
    console.log("Minified:", formatBytes(full.minified));
    console.log("Gzipped:", formatBytes(full.gz));

    console.log("\n✔ Done.");
}

main().catch((err) => {
    console.log("❌ Error:", err.message || err);
});
