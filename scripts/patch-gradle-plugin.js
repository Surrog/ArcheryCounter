#!/usr/bin/env node
// Patches @react-native/gradle-plugin to be compatible with Gradle 9.4.0+:
// 1. Bumps Kotlin compiler from 2.1.x to 2.3.0 (Gradle 9.4.0 bundles Kotlin 2.3.0 stdlib,
//    which the 2.1.x compiler cannot read — metadata version 2.3.0 > max 2.2.0).
// 2. Bumps apiVersion from KOTLIN_1_8 to KOTLIN_1_9 (Kotlin 2.3.0 dropped 1.8 API support).
const fs = require('fs');
const path = require('path');

const base = path.resolve(__dirname, '../node_modules/@react-native/gradle-plugin');

function patch(file, replacements) {
  const full = path.join(base, file);
  if (!fs.existsSync(full)) { console.log(`patch-gradle-plugin: not found: ${file}`); return; }
  let content = fs.readFileSync(full, 'utf8');
  let changed = false;
  for (const [from, to] of replacements) {
    const next = content.replace(from, to);
    if (next !== content) { content = next; changed = true; }
  }
  if (changed) { fs.writeFileSync(full, content); console.log(`patch-gradle-plugin: patched ${file}`); }
  else { console.log(`patch-gradle-plugin: already patched ${file}`); }
}

patch('gradle/libs.versions.toml', [
  [/^kotlin = "2\.\d+\.\d+"/m, 'kotlin = "2.3.0"'],
]);

const api18to19 = [[/KotlinVersion\.KOTLIN_1_8/g, 'KotlinVersion.KOTLIN_1_9']];
for (const sub of ['shared', 'shared-testutil', 'settings-plugin', 'react-native-gradle-plugin']) {
  patch(`${sub}/build.gradle.kts`, api18to19);
}
