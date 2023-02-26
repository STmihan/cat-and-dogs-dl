import * as tfjs from "@tensorflow/tfjs";

const layerPath = import.meta.env.VITE_HOST + import.meta.env.BASE_URL + "model/model.json"
const graphModel = await tfjs.loadLayersModel(layerPath);


export async function generateResult() {
    const pixels = await tfjs.browser.fromPixelsAsync(image, 3)
    const resized = tfjs.image.resizeBilinear(pixels, [224, 224]);
    const batched = resized.expandDims(0);
    return await graphModel.predict(batched).data();
}