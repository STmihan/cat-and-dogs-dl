import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import topLevelAwait from "vite-plugin-top-level-await";

// https://vitejs.dev/config/
export default ({mode}) => {
    return defineConfig({
        plugins: [vue(), topLevelAwait({
            // The export name of top-level await promise for each chunk module
            promiseExportName: "__tla", // The function to generate import names of top-level await promise in each chunk module
            promiseImportName: i => `__tla_${i}`
        })],
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
    })
}
