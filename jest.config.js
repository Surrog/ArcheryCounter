module.exports = {
  preset: 'react-native',
  testMatch: ['<rootDir>/src/__tests__/**/*.test.(ts|tsx)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  moduleNameMapper: {
    // Map the native module to our mock
    '^./NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
    '^../NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|react-native-svg|react-native-image-picker)/)',
  ],
};
