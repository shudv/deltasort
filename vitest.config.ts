import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        // Enable globals like describe, it, expect without imports
        globals: true,

        // Test environment (node for library, jsdom for browser code)
        environment: "jsdom",

        // Coverage configuration
        coverage: {
            provider: "v8",
            reporter: ["text", "json", "html"],
            thresholds: {
                statements: 100,
                branches: 100,
                functions: 100,
                lines: 100,
            },
            include: ["src/**/*.{ts,tsx}"],
            exclude: [
                "node_modules/",
                "dist/",
                "**/*.test.ts",
                "**/*.config.ts",
                "**/contracts/**",
                "**/index.{ts,tsx}",
            ],
        },

        // Test file patterns
        include: ["src/**/*.{test,spec}.{ts,tsx}"],
    },
});
