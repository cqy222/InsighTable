import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    displayMode: 'vis',
    hp_test: [],
  },
  mutations: {
    ['UPDATE_DISPLAY_MODE'](state, displayMode) {
      state.displayMode = displayMode;
    },
    // ['UPDATE_HP_TEST'](state, hp_test){
    //   state.hp_test = hp_test
    // }
  },
  actions: {

  },
  getters: {
  }
})
