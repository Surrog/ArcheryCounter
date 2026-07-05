import { describe, expect, it, beforeAll, afterAll, jest } from '@jest/globals';

// Prevent logEvent from writing to disk
jest.mock('fs', () => ({
  ...jest.requireActual<typeof import('fs')>('fs'),
  appendFileSync: jest.fn<any>().mockImplementation(() => {}),
}));

import {
  dbToTargets,
  clampBoundary,
  targetsToDB,
  isOldFormatRings,
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
    it('(null, null) → []', () => expect(dbToTargets(null, null)).toEqual([]));
    it('(null, []) → []',   () => expect(dbToTargets(null, [])).toEqual([]));
    it('([], null) → []',   () => expect(dbToTargets([], null)).toEqual([]));
    it('(undefined, []) → []', () => expect(dbToTargets(undefined, [])).toEqual([]));
    it('([], undefined) → []', () => expect(dbToTargets([], undefined)).toEqual([]));
    it('(0, []) → []',         () => expect(dbToTargets(0, [])).toEqual([]));
    it('("str", []) → []',     () => expect(dbToTargets('str', [])).toEqual([]));
    it('({}, []) → []',        () => expect(dbToTargets({}, [])).toEqual([]));
    it('([], false) → []',     () => expect(dbToTargets([], false)).toEqual([]));
  });

  describe('new format — boundary edge cases', () => {
    it('[] (empty boundary array) → []', () => {
      expect(dbToTargets([], [])).toEqual([]);
    });

    it('boundary with empty points array → skipped, returns []', () => {
      expect(dbToTargets([{ points: [] }], [])).toEqual([]);
    });

    it('boundary with all-zero points → skipped, returns []', () => {
      const zeroBoundary = [{ points: [[0,0],[0,0],[0,0],[0,0]] }];
      expect(dbToTargets(zeroBoundary, [])).toEqual([]);
    });

    it('boundary with at least one non-zero point → kept', () => {
      const almostZero = [{ points: [[0,0],[0,0],[0,0],[1,0]] }];
      expect(dbToTargets(almostZero, [])).toHaveLength(1);
    });

    it('mix: first boundary all-zero, second valid → only second kept', () => {
      const mixed = [
        { points: [[0,0],[0,0],[0,0],[0,0]] },
        boundary(500, 50),
      ];
      const result = dbToTargets(mixed, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points[0][0]).not.toBe(0);
    });
  });

  describe('new format — rings assignment', () => {
    it('valid boundary + empty rings [] → 1 target with 0 ring sets', () => {
      const result = dbToTargets([boundary(50, 50)], []);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('1 boundary + 1 ring set → assigned to that target', () => {
      const result = dbToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('2 boundaries + 2 ring sets → each ring set goes to nearest target', () => {
      const boundaries = [boundary(50, 50), boundary(500, 50)];
      const rings = [rs(45, 45), rs(500, 50)];
      const result = dbToTargets(boundaries, rings);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(1);
      // first ring set is near (50,50), not (500,50)
      expect(result[0].ringSets[0][0].points[0][0]).toBeLessThan(100);
      expect(result[1].ringSets[0][0].points[0][0]).toBeGreaterThan(400);
    });

    it('2 ring sets both near same target → both assigned to it', () => {
      const boundaries = [boundary(50, 50), boundary(500, 50)];
      const rings = [rs(40, 40), rs(60, 60)];
      const result = dbToTargets(boundaries, rings);
      expect(result[0].ringSets).toHaveLength(2);
      expect(result[1].ringSets).toHaveLength(0);
    });

    it('empty RingSet (rings=[]) → skipped, not pushed', () => {
      const result = dbToTargets([boundary(50, 50)], [[]]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('RingSet with first ring having empty points → skipped', () => {
      const emptyPointsRS = [{ points: [] as [number,number][] }];
      const result = dbToTargets([boundary(50, 50)], [emptyPointsRS]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('mix of valid and empty ring sets → only valid ones pushed', () => {
      const rings = [[], rs(50, 50), [{ points: [] as [number,number][] }], rs(60, 60)];
      const result = dbToTargets([boundary(50, 50)], rings);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('old format — boundary migration', () => {
    it('old flat [x,y][] boundary + empty rings → 1 target, 0 ring sets', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = dbToTargets(oldBoundary, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('old flat boundary + old flat SplineRing[] → 1 target, 1 ring set with all rings', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const oldRings = [ring(45, 45), ring(50, 50)]; // flat SplineRing[]
      const result = dbToTargets(oldBoundary, oldRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(2);
    });

    it('old flat boundary + new RingSet[] → 1 target, ring set assigned', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const newRings = [rs(50, 50)];
      const result = dbToTargets(oldBoundary, newRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });
  });

  describe('old format — boundary migration 20190321_212022.jpg', () => {
    it('old boundary + empty rings → 1 target, 0 ring sets', () => {
      const oldBoundary = [[[537, 299], [534, 509], [524, 752], [309, 772], [81, 774], [70, 619], [55, 307], [280, 304]]];
      const result = dbToTargets(oldBoundary, []);
      // dump generated result for debugging if test fails, since this was a real-world edge case
      console.log('generateddbToTargets output:', JSON.stringify(result, null, 2));
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary[0]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('old boundary + old ring → 1 target, 1 ring set with all rings', () => {
      const oldBoundary = [[[537, 299], [534, 509], [524, 752], [309, 772], [81, 774], [70, 619], [55, 307], [280, 304]]];
      const oldRings = [[[{"points": [[319.75, 561.9166870117188], [302.75, 568.2166748046875], [285.75, 559.2166748046875], [281.68388939644456, 540.6081073548487], [290.75, 528.9166870117188], [304.75, 523.9166870117188], [319.75, 531.5], [325.75, 546.5]]}, {"points": [[322.92582730418087, 581.8131844699905], [289.3227100829142, 584.4548110850145], [263.52553985011946, 562.418942659258], [261.75, 530.2166748046875], [283.75, 506.2166748046875], [315.75, 504.2166748046875], [339.75, 523.2166748046875], [344.65064118371436, 556.0414918585415]]}, {"points": [[334.7432145275989, 601.4692877092043], [280.75, 607.9166870117188], [243.75, 576.9166870117188], [239.43338090539368, 522.4978392393069], [273.75, 485.91668701171875], [319.75, 481.91668701171875], [359.9527647407154, 511.2297428621181], [365.863209335641, 562.1575446770947]]}, {"points": [[338.33635818984254, 624.3843651505729], [267.75, 625.5], [220.75, 579.5], [221.75, 509.2166748046875], [268.75, 464.2166748046875], [335.75, 462.2166748046875], [383.4441064994012, 507.1880025210357], [385.01483264172833, 575.5030093243266]]}, {"points": [[347.600350350042, 643.717363901502], [263.75, 647.2166748046875], [201.75, 589.2166748046875], [200.04725120237538, 502.0309195284556], [260.75, 441.91668701171875], [341.6775328765211, 439.23781524117055], [402.7960277136666, 497.9262621049075], [405.2493390386593, 582.624464387946]]}, {"points": [[379.75, 434.2166748046875], [432.75, 516.2166748046875], [410.75, 619.2166748046875], [324.75, 672.9166870117188], [225.61879280716008, 653.7941114983155], [170.18664738315226, 567.2484575038859], [191.75, 469.2166748046875], [279.75, 411.2166748046875]]}, {"points": [[393.75, 665.9166870117188], [273.75, 692.9166870117188], [176.75, 635.2166748046875], [149.48658971330514, 519.1754490635531], [207.75, 419.5], [327.75, 388.2166748046875], [425.8033829781032, 450.1785554039767], [455.8100005277295, 565.4799348528486]]}, {"points": [[421.0511927470803, 407.55424522495446], [479.97712315905795, 529.2638348180881], [432.75, 658.2166748046875], [311.75, 716.9166870117188], [184.2453974939544, 677.1011386914472], [125.3194670819767, 555.3915490983136], [167.75, 422.5], [292.75, 361.5]]}, {"points": [[434.75, 686.2166748046875], [294.75, 741.2166748046875], [151.75, 675.2166748046875], [100.75, 534.5], [165.99312529704744, 392.27366542147536], [316.75, 338.5], [452.75, 408.5], [499.75, 557.2166748046875]]}, {"points": [[444.75, 704.2166748046875], [293.75, 761.4000244140625], [136.75, 691.61669921875], [76.75, 531.2166748046875], [150.80653077319363, 375.5539462890472], [314.75, 313.5], [471.75, 391.70001220703125], [521.75, 556.2166748046875]]}]]]; // flat SplineRing[]
      expect(isOldFormatRings(oldRings)).toBe(true);
      const result = dbToTargets(oldBoundary, oldRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(10);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary[0]);
    });
  });


  describe('[RingSet] — single-element RingSet[] is valid new format', () => {
    it('[RingSet] processed correctly without unwanted flattening', () => {
      // A single-element RingSet[] must NOT be misidentified as double-nested.
      const result = dbToTargets([boundary(50, 50)], [rs(50, 50)]);
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
      const result = dbToTargets(dbBoundary, dbRings);
      expect(result).toHaveLength(2);
      expect(result[0].ringSets).toHaveLength(2);
      expect(result[1].ringSets).toHaveLength(1);
    });
  });
});

// ---- dbToTargets -----------------------------------------------------

describe('dbToTargets', () => {

  describe('null / undefined / non-array inputs', () => {
    it('(null, null) → []', () => expect(dbToTargets(null, null)).toEqual([]));
    it('(null, []) → []',   () => expect(dbToTargets(null, [])).toEqual([]));
    it('([], null) → []',   () => expect(dbToTargets([], null)).toEqual([]));
    it('(undefined, []) → []', () => expect(dbToTargets(undefined, [])).toEqual([]));
    it('(0, []) → []',         () => expect(dbToTargets(0, [])).toEqual([]));
    it('(true, []) → []',      () => expect(dbToTargets(true, [])).toEqual([]));
    it('([], "str") → []',     () => expect(dbToTargets([], 'str')).toEqual([]));
  });

  describe('new format — boundary edge cases', () => {
    it('[] → []', () => expect(dbToTargets([], [])).toEqual([]));

    it('boundary with empty points → skipped, returns []', () => {
      expect(dbToTargets([{ points: [] }], [])).toEqual([]);
    });

    it('all-zero boundary → skipped, returns []', () => {
      expect(dbToTargets([{ points: [[0,0],[0,0],[0,0],[0,0]] }], [])).toEqual([]);
    });

    it('boundary with one non-zero point → kept', () => {
      expect(dbToTargets([{ points: [[0,0],[0,0],[0,0],[0,1]] }], [])).toHaveLength(1);
    });

    it('mix: zero then valid boundary → only valid kept', () => {
      const mixed = [
        { points: [[0,0],[0,0],[0,0],[0,0]] },
        boundary(500, 50),
      ];
      const result = dbToTargets(mixed, []);
      expect(result).toHaveLength(1);
    });
  });

  describe('new format — rings assignment', () => {
    it('1 boundary + empty rings → 1 target, 0 ring sets', () => {
      const result = dbToTargets([boundary(50, 50)], []);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('1 boundary + 1 ring set → assigned', () => {
      const result = dbToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('2 boundaries + 2 ring sets → centroid matching', () => {
      const result = dbToTargets(
        [boundary(50, 50), boundary(500, 50)],
        [rs(45, 45), rs(500, 50)],
      );
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(1);
    });

    it('empty RingSet → skipped', () => {
      const result = dbToTargets([boundary(50, 50)], [[]]);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('RingSet with empty first ring points → skipped', () => {
      const result = dbToTargets(
        [boundary(50, 50)],
        [[{ points: [] as [number,number][] }]],
      );
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('mix of valid, empty, and zero-points ring sets → only valid pushed', () => {
      const rings = [[], rs(50, 50), [{ points: [] as [number,number][] }], rs(60, 60)];
      const result = dbToTargets([boundary(50, 50)], rings);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('old annotation boundary format', () => {
    it('old flat [x,y][] boundary + empty rings → 1 target', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = dbToTargets(oldBoundary, []);
      expect(result).toHaveLength(1);
      expect(result[0].paperBoundary.points).toEqual(oldBoundary);
      expect(result[0].ringSets).toHaveLength(0);
    });

    it('old flat boundary + old flat SplineRing[] → 1 target, 1 ring set', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const oldRings = [ring(45, 45), ring(55, 55)]; // isOldAnnotationRings path
      const result = dbToTargets(oldBoundary, oldRings);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(2);
    });

    it('old flat boundary + new RingSet[] → 1 target, ring set assigned', () => {
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const result = dbToTargets(oldBoundary, [rs(50, 50)]);
      expect(result).toHaveLength(1);
      expect(result[0].ringSets).toHaveLength(1);
    });

    it('old flat boundary + boundaryCentroids populated → centroid used for assignment', () => {
      // Even with old format, the centroid must be computed so all ring sets
      // reach closest_target_idx=0 (the only target). No index-out-of-bounds.
      const oldBoundary = [[0,0],[100,0],[100,100],[0,100]];
      const twoRingSets = [rs(30, 30), rs(70, 70)];
      const result = dbToTargets(oldBoundary, twoRingSets);
      expect(result[0].ringSets).toHaveLength(2);
    });
  });

  describe('[RingSet] — single-element RingSet[] is valid new format', () => {
    it('[RingSet] processed correctly without unwanted flattening', () => {
      const result = dbToTargets([boundary(50, 50)], [rs(50, 50)]);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[0].ringSets[0]).toHaveLength(1);
    });
  });

  describe('round-trip with targetsToDB', () => {
    it('targetsToDB → dbToTargets reproduces equivalent structure', () => {
      const original: TargetData[] = [
        { paperBoundary: boundary(50, 50), ringSets: [rs(45, 45)] },
        { paperBoundary: boundary(500, 50), ringSets: [rs(495, 50), rs(505, 50)] },
      ];
      const { boundary: dbBoundary, rings: dbRings } = targetsToDB(original, []);
      const result = dbToTargets(dbBoundary, dbRings);
      expect(result).toHaveLength(2);
      expect(result[0].ringSets).toHaveLength(1);
      expect(result[1].ringSets).toHaveLength(2);
    });
  });
});