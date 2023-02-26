<template>
  <div class="images">
    <div class="img_container">
      <ImageItem
          @removeImage="(id) => $emit('removeImage', id)"
          class="image_item"
          v-for="image in images"
          :key="image.id"
          :image="image"
      />
      <div class="upload_image_item image_item">
        <input
            type="file"
            id="upload_image"
            multiple
            @change="uploadFiles"
            accept="image/*"
        >
        <label for="upload_image">+</label>
      </div>
    </div>
  </div>
</template>

<script>
import ImageItem from "./ImageItem.vue";

export default {
  name: "ImageForm",
  components: {ImageItem},
  props: {
    images: {
      type: Array,
      required: true
    }
  },
  data() {
    return {
      isDrag: false
    }
  },
  methods: {
    uploadFiles(e) {
      const files = e.target.files;

      if (files.length === 0) {
        return;
      }
      const images = [];
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const reader = new FileReader();
        const id = Date.now() + i;
        reader.onload = (e) => {
          images.push({
            id: id,
            src: e.target.result
          });
          if (images.length === files.length) {
            this.$emit("newImages", images);
          }
        };
        reader.readAsDataURL(file);
      }
    }
  }
}
</script>

<style scoped>
.images {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  border-radius: 0.25rem;
  box-shadow: var(--secondary-color) 0 0 2px;
  margin: 1rem;
}

.img_container {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  padding: 1rem;
  align-content: flex-start;
}

.upload_image_item {
  border: 2px dashed var(--secondary-color);
}

.image_item {
  width: min(30%, 150px);
  aspect-ratio: 1 / 1;
  border-radius: 0.25rem;
  margin: 0.25rem;
}

.upload_image_item label {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
  width: 100%;
  cursor: pointer;
  transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
  font-size: 1.5rem;
}

.upload_image_item label:hover {
  background-color: var(--secondary-color);
  color: var(--primary-color);
}

.upload_image_item input {
  display: none;
}
</style>