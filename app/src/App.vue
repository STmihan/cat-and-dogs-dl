<template>
  <Navbar
      :links="links"
      class="navbar"
  />
  <main>
    <ImageAIModule/>
  </main>

</template>

<script>
import Navbar from "./components/Navbar.vue";
import {defineComponent} from "vue";
import ImageAIModule from "./components/ImageAiModule/ImageAIModule.vue";

export default defineComponent({
  name: "App",
  components: {
    ImageAIModule,
    Navbar,
  },
  data() {
    return {
      links: [
        {
          text: "Is a cat or a dog?",
          href: "",
          disable: false,
        }
      ],
    };
  },
});
</script>

<style scoped>
.navbar {
  height: 4rem;
}

main {
  height: calc(100vh - 4rem);
}
</style>
