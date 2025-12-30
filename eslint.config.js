import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";

export default tseslint.config(
    // Global ignores
    {
        ignores: ["dist/**", "node_modules/**", "*.cjs"],
    },

    // Base config for all files
    eslint.configs.recommended,

    // TypeScript files
    ...tseslint.configs.recommended,

    // Prettier config (disables conflicting rules)
    prettier,

    // Custom rules
    {
        rules: {
            "@typescript-eslint/no-unused-vars": [
                "error",
                { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
            ],
            "@typescript-eslint/explicit-module-boundary-types": "off",
            "@typescript-eslint/no-explicit-any": "warn",
        },
    },
);
