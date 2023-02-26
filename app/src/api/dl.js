import * as tfjs from "@tensorflow/tfjs";

const layerPath = import.meta.env.VITE_HOST + import.meta.env.BASE_URL + "model/model.json"
const graphModel = await tfjs.loadLayersModel(layerPath);


export async function generateResult(imageBase64) {
    const image = await load(imageBase64);

    const pixels = await tfjs.browser.fromPixelsAsync(image, 3);
    console.log(pixels);
    const resized = tfjs.image.resizeBilinear(pixels, [224, 224]);
    const batched = resized.expandDims(0);
    return await graphModel.predict(batched).data();
}

function load(url){
    return new Promise((resolve, reject) => {
        const im = new Image()
        im.crossOrigin = 'anonymous'
        im.src = url
        im.onload = () => {
            resolve(im)
        }
    })
}