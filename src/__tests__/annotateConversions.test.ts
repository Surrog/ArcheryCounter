import { generateddbToTargets, annotationToTargets } from '../../scripts/annotateInterface';

// ---- generateddbToTargets ----

import { test, expect } from '@jest/globals';


test('new format: single target, single ring set assigned correctly', () => {
  const boundary = [{ points: [[0,0],[100,0],[100,100],[0,100]] }];
  const rings = [[
    { points: [[50,50],[60,50],[60,60],[50,60]] },  // ring 0
  ]];
  const result = generateddbToTargets(boundary, rings);
  expect(result).toHaveLength(1);
  expect(result[0].paperBoundary.points).toHaveLength(4);
  expect(result[0].ringSets).toHaveLength(1);
  expect(result[0].ringSets[0]).toHaveLength(1);
});

test('new format: two targets, ring sets assigned to nearest target', () => {
  const boundary = [
    { points: [[0,0],[100,0],[100,100],[0,100]] },     // centroid ~(50,50)
    { points: [[500,0],[600,0],[600,100],[500,100]] },  // centroid ~(550,50)
  ];
  const ringNearFirst  = [{ points: [[40,40],[60,40],[60,60],[40,60]] }];
  const ringNearSecond = [{ points: [[540,40],[560,40],[560,60],[540,60]] }];
  const result = generateddbToTargets(boundary, [ringNearFirst, ringNearSecond]);
  expect(result[0].ringSets).toHaveLength(1);
  expect(result[1].ringSets).toHaveLength(1);
  expect(result[0].ringSets[0][0].points[0][0]).toBeCloseTo(40);
  expect(result[1].ringSets[0][0].points[0][0]).toBeCloseTo(540);
});

test('old format boundary (flat [number,number][]) is migrated', () => {
  const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
  const rings = [[{ points: [[50,50],[60,50],[60,60],[50,60]] }]];
  const result = generateddbToTargets(oldBoundary, rings);
  expect(result).toHaveLength(1);
  expect(result[0].paperBoundary.points).toHaveLength(4);
});

test('old format rings (flat SplineRing[]) is migrated into result[0].ringSets', () => {
  const boundary = [{ points: [[0,0],[100,0],[100,100],[0,100]] }];
  const oldRings = [
    { points: [[45,45],[55,45],[55,55],[45,55]] },
    { points: [[40,40],[60,40],[60,60],[40,60]] },
  ];
  const result = generateddbToTargets(boundary, oldRings);
  expect(result[0].ringSets).toHaveLength(1);
  expect(result[0].ringSets[0]).toHaveLength(2);  // both SplineRings in one RingSet
});

// ---- annotationToTargetData ----

test('new format annotation: ring sets assigned to nearest target', () => {
  const boundary = [
    { points: [[0,0],[100,0],[100,100],[0,100]] },
    { points: [[500,0],[600,0],[600,100],[500,100]] },
  ];
  const ringNearFirst  = [{ points: [[50,50],[60,50],[60,60],[50,60]] }];
  const ringNearSecond = [{ points: [[550,50],[560,50],[560,60],[550,60]] }];
  const result = annotationToTargets(boundary, [ringNearFirst, ringNearSecond]);
  expect(result[0].ringSets).toHaveLength(1);
  expect(result[1].ringSets).toHaveLength(1);
});

test('old format annotation boundary is migrated', () => {
  const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
  const rings = [[{ points: [[50,50],[60,50],[60,60],[50,60]] }]];
  const result = annotationToTargets(oldBoundary, rings);
  expect(result).toHaveLength(1);
  expect(result[0].paperBoundary.points).toEqual([[0,0],[100,0],[100,100],[0,100]]);
});
