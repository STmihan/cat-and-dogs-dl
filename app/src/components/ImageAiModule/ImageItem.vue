<template>
  <div class="item">
    <img
        :key="image.id"
        :src="image.src"
        alt="image">
    <div
        @click="_ => $emit('removeImage', image.id)"
        class="remove-item">
      Remove
    </div>
  </div>
</template>

<script>
export default {
  name: "ImageItem",
  props: {
    image: {
      required: true
    }
  }
}
</script>

<style scoped>
.item {
  position: relative;
  overflow: hidden;
}

.item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.remove-item {
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  bottom: 0;
  background-color: #000000;
  color: #ffffff;
  padding: 5px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.3s;
  display: flex;
  justify-content: center;
  align-items: center;
}

.remove-item:hover {
  opacity: 0.8;
}

</style>