// https://eslint.org/docs/user-guide/configuring

module.exports = {
  root: true,
  parser: "vue-eslint-parser",
  parserOptions: {
    sourceType: 'module',
    jsx: true,
    ecmaVersion: 2018,
    ecmaFeatures: {
      experimentalObjectRestSpread: true
    }
  },
  env: {
    browser: true,
  },
  // https://github.com/standard/standard/blob/master/docs/RULES-en.md
  extends: ['vue', 'plugin:vue/recommended'],
  plugins: [
    'html'
  ],
  // add your custom rules here
  'rules': {
    // allow paren-less arrow functions
    'arrow-parens': 0,
    // allow async-await
    'generator-star-spacing': 0,
    // allow debugger during development
    'no-debugger': process.env.NODE_ENV === 'production' ? 2 : 0,
    'camelcase': 'off',
    'no-new': 0,
    'handle-callback-err': 0,
    // ignore comma-dangle
    'comma-dangle': 0
  },
  "globals": {
    "d3": false
  },

}
