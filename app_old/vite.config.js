import topLevelAwait from "vite-plugin-top-level-await";
import {defineConfig} from "vite";

export default ({mode}) => {
    return defineConfig({
        plugins: [
            topLevelAwait({
                // The export name of top-level await promise for each chunk module
                promiseExportName: "__tla",
                // The function to generate import names of top-level await promise in each chunk module
                promiseImportName: i => `__tla_${i}`
            })
        ],
        build: {
            rollupOptions: {
                output: {
                    manualChunks(id) {
                        if (id.includes('@tensorflow')) {
                            return id.toString().split('@tensorflow/')[1].split('/')[0].toString();
                        }
                        if (id.includes('node_modules')) {
                            return id.toString().split('node_modules/')[1].split('/')[0].toString();
                        }
                    }
                }
            }
        },
        base: mode === "production" ? '/cat-and-dogs-dl/' : '/'
    });
}