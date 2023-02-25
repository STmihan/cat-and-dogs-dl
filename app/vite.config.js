import topLevelAwait from "vite-plugin-top-level-await";
import {defineConfig} from "vite";

export default defineConfig({
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
            output:{
                manualChunks(id) {
                    if (id.includes('tfjs-core')) {
                        return 'tfjs-core';
                    }
                    if (id.includes('tfjs-layers')) {
                        return 'tfjs-layers';
                    }
                    if (id.includes('node_modules')) {
                        return id.toString().split('node_modules/')[1].split('/')[0].toString();
                    }
                }
            }
        }
    },
    base: '/cat-and-dogs-dl/'
});