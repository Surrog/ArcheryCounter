module.exports = {
  preset: 'react-native',
  maxWorkers: 2,
  testMatch: ['<rootDir>/src/__tests__/**/*.test.(ts|tsx)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/__tests__/**',
    '!src/**/__mocks__/**',
    '!src/NativeArcheryCounter.ts',
  ],
  coverageReporters: ['text', 'lcov'],
  moduleNameMapper: {
    // Map the native module to our mock
    '^./NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
    '^../NativeArcheryCounter$': '<rootDir>/src/__mocks__/NativeArcheryCounter.ts',
  },
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|react-native-svg|react-native-image-picker)/)',
  ],
};
