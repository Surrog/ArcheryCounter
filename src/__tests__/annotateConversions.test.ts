import { describe, expect, it, beforeAll, afterAll, jest } from '@jest/globals';

// Prevent logEvent from writing to disk
jest.mock('fs', () => ({
  ...jest.requireActual<typeof import('fs')>('fs'),
  appendFileSync: jest.fn<any>().mockImplementation(() => {}),
}));

import {
  generateddbToTargets,
  annotationToTargets,
  clampBoundary,
  targetsToDB,
  TargetData,
  ArrowData,
} from '../../scripts/annotateInterface';

// Silence logEvent console output
beforeAll(() => {
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
});
afterAll(() => {
  jest.restoreAllMocks();
});

// ---- Fixtures ----------------------------------------------------------------

const ring = (cx: number, cy: number, r = 10) => ({
  points: [
    [cx - r, cy - r], [cx + r, cy - r],
    [cx + r, cy + r], [cx - r, cy + r],
  ] as [number, number][],
});

const boundary = (cx: number, cy: number, r = 50) => ({
  points: [
    [cx - r, cy - r], [cx + r, cy - r],
    [cx + r, cy + r], [cx - r, cy + r],
  ] as [number, number][],
});

// A valid RingSet (array of SplineRings)
const rs = (cx: number, cy: number) => [ring(cx, cy)];

// ---- clampBoundary -----------------------------------------------------------

describe('clampBoundary', () => {
  it('returns null for null input', () => {
    expect(clampBoundary(null, 100, 100)).toBeNull();
  });

  it('returns empty array for empty input', () => {
    expect(clampBoundary([], 100, 100)).toEqual([]);
  });

  it('passes through points already within bounds', () => {
    expect(clampBoundary([[10, 20], [50, 80]], 100, 100)).toEqual([[10, 20], [50, 80]]);
  });

  it('clamps negative x to 0', () => {
    expect(clampBoundary([[-5, 50]], 100, 100)).toEqual([[0, 50]]);
  });

  it('clamps negative y to 0', () => {
    expect(clampBoundary([[50, -99]], 100, 100)).toEqual([[50, 0]]);
  });

  it('clamps x >= w to w-1', () => {
    expect(clampBoundary([[100, 50]], 100, 100)).toEqual([[99, 50]]);
    expect(clampBoundary([[999, 50]], 100, 100)).toEqual([[99, 50]]);
  });

  it('clamps y >= h to h-1', () => {
    expect(clampBoundary([[50, 100]], 100, 100)).toEqual([[50, 99]]);
    expect(clampBoundary([[50, 9999]], 100, 100)).toEqual([[50, 99]]);
  });

  it('rounds float coordinates', () => {
    expect(clampBoundary([[10.6, 20.4]], 100, 100)).toEqual([[11, 20]]);
  });

  it('rounds and clamps together', () => {
    expect(clampBoundary([[99.9, -0.4]], 100, 100)).toEqual([[99, 0]]);
  });

  it('handles w=1, h=1 — all points become [0,0]', () => {
    expect(clampBoundary([[5, 5], [0, 0], [100, 100]], 1, 1)).toEqual([[0, 0], [0, 0], [0, 0]]);
  });

  it('clamps a full boundary polygon correctly', () => {
    const result = clampBoundary([[-10, -20], [1050, 30], [800, 1080], [10, 750]], 1000, 1000);
    expect(result).toEqual([[0, 0], [999, 30], [800, 999], [10, 750]]);
  });
});

// ---- targetsToDB -------------------------------------------------------------

describe('targetsToDB', () => {
  it('empty targets and arrows → all empty arrays', () => {
    const out = targetsToDB([], []);
    expect(out.boundary).toEqual([]);
    expect(out.rings).toEqual([]);
    expect(out.arrows).toEqual([]);
  });

  it('passes arrows through unchanged', () => {
    const arrows: ArrowData[] = [{ tip: [100, 200], score: 9 }, { tip: [50, 50], score: 'X' }];
    expect(targetsToDB([], arrows).arrows).toEqual(arrows);
  });

  it('1 target, 0 ring sets → boundary has 1 entry, rings is empty', () => {
    const targets: TargetData[] = [{ paperBoundary: boundary(50, 50), ringSets: [] }];
    const out = targetsToDB(targets, []);
    expect(out.boundary).toHaveLength(1);
    expect(out.rings).toHaveLength(0);
  });

  it('1 target, 1 ring set → rings has 1 entry', () => {
    const targets: TargetData[] = [{ paperBoundary: boundary(50, 50), ringSets: [rs(50, 50)] }];
    const out = targetsToDB(targets, []);
    expect(out.boundary).toHaveLength(1);
    expect(out.rings).toHaveLength(1);
    expect(out.rings[0]).toEqual(rs(50, 50));
  });

  it('1 target, 2 ring sets → rings has 2 entries (flattened)', () => {
    const targets: TargetData[] = [{
      paperBoundary: boundary(50, 50),
      ringSets: [rs(40, 40), rs(60, 60)],
    }];
    const out = targetsToDB(targets, []);
    expect(out.rings).toHaveLength(2);
  });

  it('2 targets, 1 ring set each → boundary has 2, rings has 2', () => {
    const targets: TargetData[] = [
      { paperBoundary: boundary(50,  50), ringSets: [rs(50,  50)] },
      { paperBoundary: boundary(500, 50), ringSets: [rs(500, 50)] },
    ];
    const out = targetsToDB(targets, []);
    expect(out.boundary).toHaveLength(2);
    expect(out.rings).toHaveLength(2);
  });

  it('2 targets with multiple ring sets each → rings are all flattened', () => {
    const targets: TargetData[] = [
      { paperBoundary: boundary(50,  50), ringSets: [rs(40, 40), rs(60, 60)] },
      { paperBoundary: boundary(500, 50), ringSets: [rs(490, 40)] },
    ];
    const out = targetsToDB(targets, []);
    expect(out.rings).toHaveLength(3);
  });
});

// ---- generateddbToTargets ----------------------------------------------------

describe('generateddbToTargets', () => {

  describe('null / undefined / non-array inputs', () => {
    it('(null, null) → []', () => expect(generateddbToTargets(null, null)).toEqual([]));
    it('(null, []) → []',   () => expect(generateddbToTargets(null, [])).toEqual([]));
    it('([], null) → []',   () => expect(generateddbToTargets([], null)).toEqual([]));
    it('(undefined, []) → []', () => expect(generateddbToTargets(undefined, [])).toEqual([]));
    it('([], undefined) → []', () => expect(generateddbToTargets([], undefined)).toEqual([]));
    it('(0, []) → []',         () => expect(generateddbToTargets(0, [])).toEqual([]));
    it('("str", []) → []',     () => expect(generateddbToTargets('str', [])).toEqual([]));
    it('({}, []) → []',        () => expect(generateddbToTargets({}, [])).toEqual([]));
    it('([], false) → []',     () => expect(generateddbToTargets([], false)).toEqual([]));
  });

  describe('new format — boundary edge cases', () => {
    it('[] (empty boundary array) → []', () => {
      expect(generateddbToTargets([], [])).toEqual([]);
    });

    it('boundary with empty points array → skipped, returns []', () => {
      expect(generateddbToTargets([{ points: [] }], [])).toEqual([]);
    });

    it('boundary with all-zero points → skipped, returns []', () => {
      const zeroBoundary = [{ points: [[0,0],[0,0],[0,0],[0,0]] }];
      expect(generateddbToTargets(zeroBoundary, [])).toEqual([]);
    });

    it('boundary with at least one non-zero point → kept', () => {
      const almostZero = [{ points: [[0,0],[0,0],[0,0],[1,0]] }];
      expect(generateddbToTargets(almostZero, [])).toHaveLength(1);
    });

    it('mix: first boundary all-zero, second valid → only second kept', () => {
      const mixed = [
        { points: [[0,0],[0,0],[0,0],[0,0]] },
        boundary(500, 50),
      ];
      const result = generateddbToTargets(mixed, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points[0][0]).not.toBe(0);
    });
  });

  describe('new format — rings assignment', () => {
    it('valid boundary + empty rings [] → 1 target with 0 ring sets', () => {
      const result = generateddbToTargets([boundary(50, 50)], []);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('1 boundary + 1 ring set → assigned to that target', () => {
      const result = generateddbToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('2 boundaries + 2 ring sets → each ring set goes to nearest target', () => {
      const boundaries = [boundary(50, 50), boundary(500, 50)];
      const rings = [rs(45, 45), rs(500, 50)];
      const result = generateddbToTargets(boundaries, rings);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(1);
      // first ring set is near (50,50), not (500,50)
      expect(result[0].ringSets[0][0].points[0][0]).toBeLessThan(100);
      expect(result[1].ringSets[0][0].points[0][0]).toBeGreaterThan(400);
    });

    it('2 ring sets both near same target → both assigned to it', () => {
      const boundaries = [boundary(50, 50), boundary(500, 50)];
      const rings = [rs(40, 40), rs(60, 60)];
      const result = generateddbToTargets(boundaries, rings);
      expect(result[0].ringSets).toHaveLength(2);
      expect(result[1].ringSets).toHaveLength(0);
    });

    it('empty RingSet (rings=[]) → skipped, not pushed', () => {
      const result = generateddbToTargets([boundary(50, 50)], [[]]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('RingSet with first ring having empty points → skipped', () => {
      const emptyPointsRS = [{ points: [] as [number,number][] }];
      const result = generateddbToTargets([boundary(50, 50)], [emptyPointsRS]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('mix of valid and empty ring sets → only valid ones pushed', () => {
      const rings = [[], rs(50, 50), [{ points: [] as [number,number][] }], rs(60, 60)];
      const result = generateddbToTargets([boundary(50, 50)], rings);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('old format — boundary migration', () => {
    it('old flat [x,y][] boundary + empty rings → 1 target, 0 ring sets', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = generateddbToTargets(oldBoundary, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('old flat boundary + old flat SplineRing[] → 1 target, 1 ring set with all rings', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const oldRings = [ring(45, 45), ring(50, 50)]; // flat SplineRing[]
      const result = generateddbToTargets(oldBoundary, oldRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(2);
    });

    it('old flat boundary + new RingSet[] → 1 target, ring set assigned', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const newRings = [rs(50, 50)];
      const result = generateddbToTargets(oldBoundary, newRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });
  });

  describe('[RingSet] — single-element RingSet[] is valid new format', () => {
    it('[RingSet] processed correctly without unwanted flattening', () => {
      // A single-element RingSet[] must NOT be misidentified as double-nested.
      const result = generateddbToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(1); // one SplineRing in the set
    });
  });

  describe('round-trip with targetsToDB', () => {
    it('targetsToDB → generateddbToTargets reproduces equivalent structure', () => {
      const original: TargetData[] = [
        { paperBoundary: boundary(50,  50), ringSets: [rs(45, 45), rs(55, 55)] },
        { paperBoundary: boundary(500, 50), ringSets: [rs(500, 50)] },
      ];
      const { boundary: dbBoundary, rings: dbRings } = targetsToDB(original, []);
      const result = generateddbToTargets(dbBoundary, dbRings);
      expect(result).toHaveLength(2);
      expect(result[0].ringSets).toHaveLength(2);
      expect(result[1].ringSets).toHaveLength(1);
    });
  });
});

// ---- annotationToTargets -----------------------------------------------------

describe('annotationToTargets', () => {

  describe('null / undefined / non-array inputs', () => {
    it('(null, null) → []', () => expect(annotationToTargets(null, null)).toEqual([]));
    it('(null, []) → []',   () => expect(annotationToTargets(null, [])).toEqual([]));
    it('([], null) → []',   () => expect(annotationToTargets([], null)).toEqual([]));
    it('(undefined, []) → []', () => expect(annotationToTargets(undefined, [])).toEqual([]));
    it('(0, []) → []',         () => expect(annotationToTargets(0, [])).toEqual([]));
    it('(true, []) → []',      () => expect(annotationToTargets(true, [])).toEqual([]));
    it('([], "str") → []',     () => expect(annotationToTargets([], 'str')).toEqual([]));
  });

  describe('new format — boundary edge cases', () => {
    it('[] → []', () => expect(annotationToTargets([], [])).toEqual([]));

    it('boundary with empty points → skipped, returns []', () => {
      expect(annotationToTargets([{ points: [] }], [])).toEqual([]);
    });

    it('all-zero boundary → skipped, returns []', () => {
      expect(annotationToTargets([{ points: [[0,0],[0,0],[0,0],[0,0]] }], [])).toEqual([]);
    });

    it('boundary with one non-zero point → kept', () => {
      expect(annotationToTargets([{ points: [[0,0],[0,0],[0,0],[0,1]] }], [])).toHaveLength(1);
    });

    it('mix: zero then valid boundary → only valid kept', () => {
      const mixed = [
        { points: [[0,0],[0,0],[0,0],[0,0]] },
        boundary(500, 50),
      ];
      const result = annotationToTargets(mixed, []);
      expect(result).toHaveLength(1);
    });
  });

  describe('new format — rings assignment', () => {
    it('1 boundary + empty rings → 1 target, 0 ring sets', () => {
      const result = annotationToTargets([boundary(50, 50)], []);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('1 boundary + 1 ring set → assigned', () => {
      const result = annotationToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('2 boundaries + 2 ring sets → centroid matching', () => {
      const result = annotationToTargets(
        [boundary(50, 50), boundary(500, 50)],
        [rs(45, 45), rs(500, 50)],
      );
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(1);
    });

    it('empty RingSet → skipped', () => {
      const result = annotationToTargets([boundary(50, 50)], [[]]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('RingSet with empty first ring points → skipped', () => {
      const result = annotationToTargets(
        [boundary(50, 50)],
        [[{ points: [] as [number,number][] }]],
      );
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('mix of valid, empty, and zero-points ring sets → only valid pushed', () => {
      const rings = [[], rs(50, 50), [{ points: [] as [number,number][] }], rs(60, 60)];
      const result = annotationToTargets([boundary(50, 50)], rings);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('old annotation boundary format', () => {
    it('old flat [x,y][] boundary + empty rings → 1 target', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = annotationToTargets(oldBoundary, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('old flat boundary + old flat SplineRing[] → 1 target, 1 ring set', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const oldRings = [ring(45, 45), ring(55, 55)]; // isOldAnnotationRings path
      const result = annotationToTargets(oldBoundary, oldRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(2);
    });

    it('old flat boundary + new RingSet[] → 1 target, ring set assigned', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = annotationToTargets(oldBoundary, [rs(50, 50)]);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('old flat boundary + boundaryCentroids populated → centroid used for assignment', () => {
      // Even with old format, the centroid must be computed so all ring sets
      // reach closest_target_idx=0 (the only target). No index-out-of-bounds.
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const twoRingSets = [rs(30, 30), rs(70, 70)];
      const result = annotationToTargets(oldBoundary, twoRingSets);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('[RingSet] — single-element RingSet[] is valid new format', () => {
    it('[RingSet] processed correctly without unwanted flattening', () => {
      const result = annotationToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(1);
    });
  });

  describe('round-trip with targetsToDB', () => {
    it('targetsToDB → annotationToTargets reproduces equivalent structure', () => {
      const original: TargetData[] = [
        { paperBoundary: boundary(50, 50), ringSets: [rs(45, 45)] },
        { paperBoundary: boundary(500, 50), ringSets: [rs(495, 50), rs(505, 50)] },
      ];
      const { boundary: dbBoundary, rings: dbRings } = targetsToDB(original, []);
      const result = annotationToTargets(dbBoundary, dbRings);
      expect(result).toHaveLength(2);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(2);
    });
  });
});