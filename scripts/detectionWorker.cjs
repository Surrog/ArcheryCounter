const { register } = require('tsx/cjs/api');
register();

const { parentPort, workerData } = require('worker_threads');
const { findTarget } = require('../src/targetDetection');

const { rgba, width, height } = workerData;
const result = findTarget(new Uint8Array(rgba), width, height);
parentPort.postMessage(result);
