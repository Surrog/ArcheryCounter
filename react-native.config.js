module.exports = {
  dependencies: {
    // Register the local native module so `pod install` and auto-linking pick it up.
    ArcheryCounterNative: {
      root: __dirname,
    },
  },
};
