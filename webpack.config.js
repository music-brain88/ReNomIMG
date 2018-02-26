var webpack = require('webpack');
module.exports = {
  entry: {
    index: './src/index.js'
  },
  output: {
    path: __dirname + '/build/',
    filename: 'build.js'
  },
  module: {
    rules: [
      {
        test: /\.vue$/,
        loader: 'vue-loader',
        options: {
          loaders: {
            'scss': 'vue-style-loader!css-loader!sass-loader',
            'sass': 'vue-style-loader!css-loader!sass-loader?indentedSyntax'
          }
        }
      },
      {
        test: /\.js$/,
        loader: 'babel-loader',
        exclude: /node_modules/
      },
      {
        test: /\.(png|jpg|gif|svg)$/,
        loader: 'file-loader',
        options: {
          name: '[name].[ext]?[hash]'
        }
      }
    ]
  },

  resolve: {
     extensions: ['.js', '.vue'],
     modules: [
         "node_modules"
     ],
     alias: {
         // vue.js のビルドを指定する
         vue: 'vue/dist/vue.common.js'
     }
  }
};
