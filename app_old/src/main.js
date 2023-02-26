import {generateResult} from "./dl.js";

const imageContainer = document.getElementById('image_container');
const uploadForm = document.getElementById('upload_form');
const resultContainer = document.getElementById('result');

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

uploadForm.onload = () => {
    generateResult().then(setResult);
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

function setResult(result) {
    let resultString = '';
    if (result[0] > result[1]) {
        resultString = `Dog ${result[0] * 100}%`;
    } else {
        resultString = `Cat ${result[1] * 100}%`;
    }
    resultContainer.innerHTML = resultString;
}