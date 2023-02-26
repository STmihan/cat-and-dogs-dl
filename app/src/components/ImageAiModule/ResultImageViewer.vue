<template>
  <div class="view">
    <div class="view_container">
      <div class="view_item" v-for="image in images" :key="image.id">
        <img :src="image.src" alt="image">
        <div v-if="results[image.id]" class="view_item_overlay">
          <p>{{ results[image.id] }}</p>
        </div>
        <div class="process_button" v-if="!results[image.id]" @click="processImage(image)">Process</div>
      </div>
    </div>
  </div>
</template>

<script>
import {generateResult} from "../../api/dl.js";

export default {
  name: "ResultImageViewer",
  props: {
    images: {
      type: Array,
      required: true
    }
  },
  data() {
    return {
      results: {}
    }
  },
  methods: {
    async processImage(image) {
      const result = await generateResult(image.src);
      console.log(result);
      this.results[image.id] = result[0] > result[1] ?
          `Dog ${Math.round(result[0] * 100)}%` : `Cat ${Math.round(result[1] * 100)}%`;
    }
  }
}
</script>

<style scoped>
.view {
  display: flex;
  margin-top: 1rem;
}

.view_container {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-start;
  align-content: flex-start;
  margin: 0 1rem;
}

.view_item {
  position: relative;
  height: min(300px, 30%);
  aspect-ratio: 1/1;
  margin-right: 0.1rem;
  margin-bottom: 0.1rem;
}

.view_item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.view_item .process_button {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: black;
  opacity: 0;
  transition: opacity 0.5s;
}

.view_item:hover .process_button {
  opacity: 0.5;
}

.view_item_overlay {
  position: absolute;
  top: 0;
  left: 0;
  height: 2rem;
  width: 100%;
  background-color: black;
  opacity: 0.8;
  display: flex;
  justify-content: center;
  align-items: center;
}

.view_item_overlay p {
  color: white;
  margin: 0;
  padding: 0;
}

@media (max-width: 600px) {
  .view {
    flex-direction: column;
  }

  .view_container {
    margin: 0 0 0 1rem;
  }

  .view_item {
    height: min(550px, 90%);
  }
}

</style>