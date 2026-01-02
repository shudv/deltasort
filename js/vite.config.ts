import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    root: "examples/react",
    server: {
        port: 3000,
        open: true,
    },
    resolve: {
        alias: {
            // Allow importing from src/ during development
            memotable: "/src/index.ts",
        },
    },
});
