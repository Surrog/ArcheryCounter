#!/usr/bin/env node
// Patches node_modules to be compatible with Gradle 9.4.0+:
// 1. @react-native/gradle-plugin: bumps Kotlin compiler 2.1.x → 2.3.0 and apiVersion 1.8 → 1.9.
// 2. onnxruntime-react-native: replaces VersionNumber (removed in Gradle 8) with plain Groovy comparison.
const fs = require('fs');
const path = require('path');

const base = path.resolve(__dirname, '../node_modules/@react-native/gradle-plugin');
const onnxBase = path.resolve(__dirname, '../node_modules/onnxruntime-react-native');

function patchFile(fullPath, replacements) {
  if (!fs.existsSync(fullPath)) { console.log(`patch-gradle-plugin: not found: ${fullPath}`); return; }
  let content = fs.readFileSync(fullPath, 'utf8');
  let changed = false;
  for (const [from, to] of replacements) {
    const next = content.replace(from, to);
    if (next !== content) { content = next; changed = true; }
  }
  if (changed) { fs.writeFileSync(fullPath, content); console.log(`patch-gradle-plugin: patched ${fullPath}`); }
  else { console.log(`patch-gradle-plugin: already patched ${fullPath}`); }
}

function patch(file, replacements) {
  patchFile(path.join(base, file), replacements);
}

patch('gradle/libs.versions.toml', [
  [/^kotlin = "2\.\d+\.\d+"/m, 'kotlin = "2.3.0"'],
]);

const api18to19 = [[/KotlinVersion\.KOTLIN_1_8/g, 'KotlinVersion.KOTLIN_1_9']];
for (const sub of ['shared', 'shared-testutil', 'settings-plugin', 'react-native-gradle-plugin']) {
  patch(`${sub}/build.gradle.kts`, api18to19);
}

// onnxruntime-react-native uses VersionNumber (internal Gradle API removed in Gradle 8+).
// Replace with a plain tokenize-based comparison; the guarded deps are only needed for RN < 0.71.
patchFile(path.join(onnxBase, 'android/build.gradle'), [
  [
    /if \(VersionNumber\.parse\(REACT_NATIVE_VERSION\) < VersionNumber\.parse\("0\.71"\)\)/,
    'if (REACT_NATIVE_VERSION.tokenize(\'.\').collect { it.toInteger() }.with { it[0] == 0 && it[1] < 71 })',
  ],
]);
