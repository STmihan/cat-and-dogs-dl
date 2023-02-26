<template>
  <div class="module">
    <ImageForm
        @newImages="newImagesHandler"
        @removeImage="removeImageHandler"
        :images="images"
        class="module_block_left"
    />
    <ResultImageViewer
        :images="images"
        class="module_block_right"
    />
  </div>
</template>

<script>
import ImageForm from "./ImageForm.vue";
import ResultImageViewer from "./ResultImageViewer.vue";

export default {
  name: "ImageAIModule",
  components: {ResultImageViewer, ImageForm},
  data() {
    return {
      images: []
    }
  },
  methods: {
    newImagesHandler(images) {
      this.images.push(...images)
      console.log(this.images);
    },
    removeImageHandler(id) {
      this.images = this.images.filter(image => image.id !== id);
    }
  }
}
</script>

<style scoped>
.module {
  display: flex;
  flex-direction: Row;
}

.module_block_left {
  width: 30%;
}

.module_block_right {
  width: 70%;
}

@media (max-width: 768px) {
  .module {
    flex-direction: column;
  }

  .module_block_left {
    width: 90%;
  }

  .module_block_right {
    width: 90%;
  }
}
</style>