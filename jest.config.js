module.exports = {
  preset: 'react-native',
  maxWorkers: 2,
  globalSetup: '<rootDir>/src/__tests__/globalSetup.ts',
  testMatch: ['<rootDir>/src/__tests__/**/*.test.(ts|tsx)'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/__tests__/**',
    '!src/**/__mocks__/**',
  ],
  coverageReporters: ['text', 'lcov'],
  moduleNameMapper: {},
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|react-native-svg|react-native-image-picker)/)',
  ],
};
