import * as core from '@tensorflow/tfjs-core';
import * as layers from "@tensorflow/tfjs-layers";

const imageContainer = document.getElementById('image_container');
const uploadForm = document.getElementById('upload_form');
const processButton = document.getElementById('process_button');
const resultContainer = document.getElementById('result');

const graphModel = await layers.loadLayersModel("http://localhost:5173/model/model.json");

let image = null;

uploadForm.onsubmit = (e) => {
    e.preventDefault();
    console.log('submit');
    const file = document.getElementById('file').files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        const imgElement = createImageIfNeed();
        imgElement.src = e.target.result;
        image = imgElement;
    };
    reader.readAsDataURL(file);
}

processButton.onclick = () => {
    generateResult().then((result) => {
        let resultString = '';
        if (result[0] > result[1]) {
            resultString = `Dog ${result[0] * 100}%`;
        } else {
            resultString = `Cat ${result[1] * 100}%`;
        }
        resultContainer.innerHTML = resultString;
    });
}

async function generateResult() {
    const pixels = await core.browser.fromPixelsAsync(image, 3)
    const resized = core.image.resizeBilinear(pixels, [224, 224]);
    const batched = resized.expandDims(0);
    const result = await graphModel.predict(batched).data();
    console.log(result);
    return result;
}

function createImageIfNeed() {
    const img = document.getElementById('image');
    if (!img) {
        const image = new Image();
        image.id = 'image';
        imageContainer.appendChild(image);
        return image;
    }
    return img;
}