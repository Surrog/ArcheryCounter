module.exports = {
  preset: 'react-native',
  maxWorkers: 2,
  testMatch: ['<rootDir>/src/__tests__/**/*.test.(ts|tsx)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  moduleNameMapper: {
    // Map the native module to our mock
    '^./NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
    '^../NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
    // Map ArcheryCounter to our mock (for useArcheryScorer tests)
    '^./ArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
    '^../ArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|react-native-svg|react-native-image-picker)/)',
  ],
};
