module.exports = {
  apps: [
    {
      name: "playpowerlabs-server",
      script: "./dist/index.js",

      env: {
        NODE_ENV: "development",
      },
    },
  ],
};
