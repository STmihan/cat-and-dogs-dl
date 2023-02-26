<template>
  <nav>
    <div class="logo">
      AI
    </div>
    <div class="links">
      <div v-for="link in links" class="link">
        <a
            v-bind:class="(link.disable) ? 'disable' : ''"
            :href="link.href"
        >{{ link.text }}</a>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: "Navbar",
  props: {
    links: {
      type: Array,
    }
  },
}
</script>

<style scoped>

  nav {
    display: flex;
    justify-content: space-between;
    flex-direction: row;
    align-items: center;
    padding-left: 3rem;
    padding-right: 1rem;
    background-color: var(--secondary-color);
    color: var(--primary-color);
  }

  .logo {
    font-size: 2rem;
  }

  .links {
    display: flex;
    justify-content: end;
    flex-direction: row;
    align-items: center;
  }

  .link {
    padding: 0.5rem;
  }

  /*noinspection ALL*/
  .disable {
    color: #ccc;
    pointer-events: none;
  }
</style>