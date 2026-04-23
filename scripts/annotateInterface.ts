import { TargetBoundary, isTargetBoundary } from '../src/targetDetection';
import { isSplineRing, RingSet, splineCentroid, isRingSet } from '../src/spline';
import * as fs from 'fs';
import * as path from 'path';

export const LOG_PATH = path.resolve(__dirname, '../logs/annotate.log');

export function logEvent(level: 'info' | 'warn' | 'error', event: string, filename: string, detail = '') {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, filename, detail });
  fs.appendFileSync(LOG_PATH, line + '\n');
  if (level === 'warn')  console.warn(`[warn]  ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
  if (level === 'error') console.error(`[error] ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
}


export interface TargetData {
  paperBoundary: TargetBoundary;
  ringSets: RingSet[];
}

export function isTargetData(x: any): x is TargetData {
  return typeof x === "object" && x != null && 
    isTargetBoundary((x as TargetData).paperBoundary) &&
    Array.isArray((x as TargetData).ringSets) &&
    (x as TargetData).ringSets.every(rs => rs.every(r => isSplineRing(r)));
}

export interface ArrowData {
  tip: [number, number]
  score: number | 'X'
}

export interface ImageData {
  filename: string;
  base64: string;
  width: number;
  height: number;
  generated: {
    targets: TargetData[];
    arrows: ArrowData[];
  };
  annotated: {
    targets: TargetData[];
    arrows: ArrowData[];
  };
}

/** True if a boundary polygon is non-degenerate (non-empty, non-null, not all-zero). */
export function isBoundaryValid(pts: TargetBoundary): boolean {
  return pts.points.length >= 3 &&
    pts.points.every(p => p[0] != null && p[1] != null) &&
    pts.points.some(p => p[0] !== 0 || p[1] !== 0);                   // at least one non-zero vertex
}

// Note: right now duplicates of annotation / generated format: here to handle case where we ever diverge the two formats in the future, and to keep the type guards simple 
function isOldFormatBoundary(pts: any): boolean {
  return Array.isArray(pts) && pts.length > 0 &&
  Array.isArray(pts[0]) && pts[0].length === 2 &&
  typeof pts[0]?.[0] === 'number' && typeof pts[0]?.[1] === 'number';
}

function isOldFormatRings(rings: any): boolean {
  return Array.isArray(rings) && rings.length > 0 &&
    rings.every(r => isSplineRing(r));
}

function isBoundaryRingsArrayPair(rawBoundary: any, rawRings: any): rawBoundary is any[] {
  return Array.isArray(rawBoundary) && Array.isArray(rawRings);
}

/** Convert DB format ex: 
 * old format: [[[529, 230], [574, 276], [549, 638], [496, 739], [97, 745], [51, 701], [8, 320], [49, 242]]]	[[[{"points": [[317.78895751251116, 512.155116841257], [313.2275451647999, 527.5573404446418], [300.123790444337, 537.0707728514956], [284.01566896411026, 536.8151611255315], [271.30574465375895, 527.2108641597003], [267.57416425538173, 512.155116841257], [273.7007596941402, 498.83944980486314], [285.16495409210086, 491.0322084749196], [299.2085657495854, 490.0562328068718], [312.0251603030828, 497.6264769749396]]}, {"points": [[343.4622351675308, 512.155116841257], [334.80010697590683, 543.2307240384354], [308.3636469637962, 562.4304436101214], [275.6097547201046, 562.6859050092147], [250.8626462098797, 542.0636445833788], [243.80809239643702, 512.155116841257], [255.50650877719806, 485.62055274850417], [278.2758157128545, 469.82962069979106], [306.41735850346464, 467.8698500253494], [332.0573129154203, 483.07226617458264]]}, {"points": [[368.01655103515196, 512.155116841257], [353.7507852142809, 556.9991977131598], [315.37753452667476, 584.0169698939909], [268.7110826706087, 583.9178344043194], [230.92164030010406, 556.5516334280371], [217.9508427639218, 512.155116841257], [234.1786922080015, 470.1249869814855], [270.3555157136608, 445.4534437827833], [314.5179703891795, 442.93873018363854], [352.64781358522436, 468.11239176504716]]}, {"points": [[392.7550231611233, 512.155116841257], [372.9984726571922, 570.9834612061896], [322.6102563844269, 606.2769988845623], [261.42184759531756, 606.3517931941432], [211.23626192351963, 570.8538979985028], [192.70627590024347, 512.155116841257], [213.12229165744887, 454.8266164947935], [262.34188857760074, 420.7900354730704], [322.5586548122715, 418.1920481070666], [373.485189778703, 452.9731517884385]]}, {"points": [[417.87494517256766, 512.155116841257], [391.9375730874133, 584.7435231109099], [329.82782622802193, 628.4903947706073], [253.8305804511231, 629.71531111013], [190.63459321890113, 585.8218864602853], [166.03402529923426, 512.155116841257], [190.66609996544088, 438.51123821350905], [253.54285744858882, 393.7094022242179], [331.1231166293886, 391.83334496775933], [395.01477745170615, 437.33099073358164]]}, {"points": [[442.9997196252475, 512.155116841257], [411.4913948231174, 598.9502061869346], [337.2186178660157, 651.2369125215531], [246.14940332770271, 653.3555434890079], [170.04935471894544, 600.7779376796364], [138.9035129853115, 512.155116841257], [168.28690045853293, 422.2517980290238], [244.33444419144578, 365.36882033920244], [339.69526451936235, 365.4509865285555], [417.117844739088, 421.27217234993464]]}, {"points": [[465.9935160440978, 512.155116841257], [428.2533738348362, 611.1284967924815], [341.67909545238683, 664.9648509570669], [239.20240282905635, 674.7362125564401], [150.24253394607712, 615.1684353157052], [113.32127586492405, 512.155116841257], [146.54717650802337, 406.4569640318819], [236.2543379964084, 340.50081052410815], [347.9379333028438, 340.08266051124633], [437.07506113155375, 406.77240590020256]]}, {"points": [[490.55371227533317, 512.155116841257], [447.65000977480463, 625.2209777031059], [350.4190781655502, 691.863751868566], [232.0019260029063, 696.8970015440941], [130.49703524511878, 629.5143798586261], [87.09732455906067, 512.155116841257], [124.39088659615368, 390.3594771480924], [227.77261254841451, 314.3967437459769], [356.42515013022233, 313.9616930051866], [458.5527131196098, 391.167978329181]]}, {"points": [[513.6521046182997, 512.155116841257], [464.27219703082864, 637.297703653076], [355.1989255836233, 706.5746095773792], [224.86016524146865, 718.877081066015], [110.53774253881261, 644.0156548386648], [62.09107710372453, 512.155116841257], [103.11366376404666, 374.90066988271997], [219.60512342818518, 289.2597969405891], [364.54977524490545, 288.9566680440057], [478.6334765440133, 376.5784497065372]]}, {"points": [[540.0777640321483, 512.155116841257], [485.49417163351654, 652.7163707301784], [364.1830420859712, 734.2248770327199], [217.9856999396448, 740.0345097523207], [90.03775517118848, 658.9097674848164], [36.08045504537941, 512.155116841257], [80.83439544720827, 358.71383395769453], [211.10202163642955, 263.08994054117744], [373.03394877003615, 262.8450668591728], [501.1958079703438, 360.18595639435625]]}]]]	[]	2026-04-15 11:40:00.167 +0200	583	1200
 * new format: rawBoundary: TargetBoundary[], rawRings: RingSet[]
 * → TargetData[] 
*/
export function generateddbToTargets(rawBoundary: any, rawRings: any): TargetData[] {
  let result = [] as TargetData[];
  let targetsCentroid = [];

  if (rawBoundary == null || rawRings == null || !isBoundaryRingsArrayPair(rawBoundary, rawRings)) {
    logEvent('error', 'missing_boundary_or_rings', 'null', 'cannot convert to TargetData: missing or non-array boundary/rings');
    return result;
  }

  if (isOldFormatBoundary(rawBoundary)) {
    logEvent('warn', 'old_format_boundary', 'null', 'migrating old boundary format to new multi-target format');
    let target = {
      paperBoundary: { points: rawBoundary as [number, number][] },
      ringSets: [] as RingSet[],
    } as TargetData;
    result.push(target);
    targetsCentroid.push(splineCentroid(target.paperBoundary));
  } else {
    for (const boundary of rawBoundary as TargetBoundary[]) {
      if (boundary.points.length === 0 || boundary.points.every(p => p[0] === 0 && p[1] === 0)) {
        logEvent('warn', 'empty_boundary', 'null', 'skipping empty boundary during migration');
        continue;
      }
      let target = {
        paperBoundary: boundary as TargetBoundary,
        ringSets: [] as RingSet[],
      } as TargetData;
      result.push(target);
      targetsCentroid.push(splineCentroid(target.paperBoundary));
    }
  }

  if (result.length === 0) {
    logEvent('error', 'no_valid_targets', 'null', 'no valid targets found during migration');
    return result;
  }

  if (isOldFormatRings(rawRings)) {
    logEvent('warn', 'old_format_rings', 'null', 'migrating old rings format to new multi-target format');
    result[0].ringSets = [rawRings as RingSet];
  } else {
    for (const rings of rawRings as RingSet[]) {
      let closest_target_idx = 0;
      if (rings.length == 0 || rings[0].points.length === 0) {
        logEvent('warn', 'empty_ring_set', 'null', 'skipping empty ring set during migration');
        continue;
      }
      const centroid = splineCentroid(rings[0]);
      let max_dist = Infinity;
      for (let i = 0; i < targetsCentroid.length; i++) {
        // find the closest target centroid for this ring set
        const dist = Math.hypot(centroid[0] - targetsCentroid[i][0], centroid[1] - targetsCentroid[i][1]);
        if (dist < max_dist) {
          max_dist = dist;
          closest_target_idx = i;
        }  
      }
      result[closest_target_idx].ringSets.push(rings);
    }
  }

  return result;
}

/** Clamp boundary points to image dimensions. */
export function clampBoundary(
  pts: [number, number][] | null,
  w: number,
  h: number,
): [number, number][] | null {
  if (!pts) return null;
  return pts.map(([x, y]) => [
    Math.round(Math.max(0, Math.min(w - 1, x))),
    Math.round(Math.max(0, Math.min(h - 1, y))),
  ]);
}

/** Convert TargetData[] → DB format { boundary, rings, arrows }. */
export function targetsToDB(
  targets: TargetData[],
  arrows: ArrowData[],
): { boundary: TargetBoundary[]; rings: RingSet[]; arrows: ArrowData[] } {
  return {
    boundary: targets.map(t => t.paperBoundary ?? ({ points: [] } as TargetBoundary)),
    rings:    targets.map(t => t.ringSets ?? []).flat(),
    arrows,
  };
}

function isOldAnnotationBoundary(pts: any): boolean {
  return Array.isArray(pts) && pts.length > 0 &&
    Array.isArray(pts[0]) && pts[0].length === 2 &&
    typeof pts[0]?.[0] === 'number' && typeof pts[0]?.[1] === 'number';
}

function isOldAnnotationRings(rings: any): boolean {
  return isRingSet(rings);
}


/* Convert DB format ex:
 * old format rawBoundary: [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]	rawRings: [[[{"points": [[457.30871071476906, 571.6529817201078], [453.86854999412947, 593.4868550286966], [435.31679070577894, 607.0463091200133], [412.49682012154534, 606.4923021061218], [394.34928520452786, 593.0623855167768], [388.5098006123553, 571.6529817201078], [398.1278187510277, 552.988843238466], [415.06058755293225, 544.7041261508197], [432.5846108956093, 544.6684391425641], [447.8855782083494, 554.1659918577445]]}, {"points": [[491.0746265153625, 571.6529817201078], [483.99711763454275, 615.3765407273429], [446.89561364582937, 642.6822618624738], [401.0783668957828, 641.6346876190569], [364.9654258251155, 614.411008992849], [353.665203087992, 571.6529817201078], [372.9460075007156, 534.6931864329105], [406.1793100437917, 517.370364571853], [441.40568310771897, 517.5199704151199], [472.1361990210538, 536.5468845067828]]}, {"points": [[528.1876967239184, 571.6529817201078], [512.1015213336848, 635.7955852390007], [456.6105730937149, 672.5818326195562], [391.6791295207504, 670.5625657501962], [338.57692011349127, 633.5833806428565], [318.14301267354597, 571.6529817201078], [341.7045239887297, 511.9949200234744], [394.4371492163758, 481.231709502451], [452.64663982009836, 482.92386299933014], [501.87857773707486, 514.937781485552]]}, {"points": [[565.4988880107312, 571.6529817201078], [540.3053120753408, 656.2868386637778], [466.27541230873766, 702.3271491610776], [382.29069583260724, 699.4571935520561], [312.04573121059906, 652.8594176993515], [282.18995596190814, 571.6529817201078], [311.50876229197144, 490.0564149852642], [382.4480108031212, 444.3329355830616], [464.02630061184334, 447.90086832193793], [531.7697933933273, 493.22054209750496]]}, {"points": [[603.1192344351948, 571.6529817201078], [567.8902162978543, 676.3284447123884], [475.6146685113878, 731.0704242254358], [373.40895647073995, 726.7923765675564], [287.4087619451176, 670.7592236318849], [247.31594008950717, 571.6529817201078], [279.7152435756911, 466.9570715229521], [370.2746488543679, 406.8671799213082], [476.32002592134825, 410.06467232631985], [563.8913231211919, 469.8828846856228]]}, {"points": [[641.3014576450207, 571.6529817201078], [595.920610973559, 696.6937185210629], [484.956323871505, 759.8210831372332], [364.6109571109527, 753.8698343572521], [262.1175913127149, 689.1343346793656], [210.48371649436643, 571.6529817201078], [248.24776107240675, 444.09460723505146], [357.97992847099346, 369.02792140322356], [488.786828365769, 371.6957996819099], [596.2566394414232, 446.3681059466288]]}, {"points": [[677.3820066499252, 571.6529817201078], [620.7628051606512, 714.742629086953], [493.9486253684132, 787.4965414155843], [355.3391946822307, 782.4053849447298], [237.7477701861023, 706.8400461277332], [175.24014133977445, 571.6529817201078], [217.02533311753146, 421.4101854982511], [345.1663183446101, 329.5916844654713], [501.61669898326954, 332.20951809834], [628.8965830897062, 422.65379877445287]]}, {"points": [[714.450074255263, 571.6529817201078], [646.0429550028542, 733.1097330616616], [503.50887490524457, 816.9199640263774], [345.98207988127297, 811.2036231230962], [213.58080585743105, 724.3983734853014], [140.1424713728702, 571.6529817201078], [185.0289001690232, 398.1634162166879], [332.21259607125694, 289.724226679632], [514.7115087051111, 291.9078377949859], [662.0836627169866, 398.5419740449332]]}, {"points": [[751.5317467638911, 571.6529817201078], [673.8357790109557, 753.3024016769157], [512.8158045972668, 845.563748321162], [336.5905467516034, 840.1077900251162], [186.91562729402486, 743.771759728473], [106.53204442679248, 571.6529817201078], [155.4759006045294, 376.6919052029593], [320.27283070904224, 252.97740738660855], [526.3876215263191, 255.97245758695294], [692.9781637407375, 376.0958051696731]]}, {"points": [[789.5736813681051, 571.6529817201078], [701.7904886946785, 773.6126871201835], [522.7671160635089, 876.1907357941182], [326.9831717952573, 869.6762497637326], [159.8655506462996, 763.4247907988499], [72.73764628574139, 571.6529817201078], [123.58796486957948, 353.52396376121624], [308.13390669270143, 215.61764078249513], [538.405578356426, 218.98498970045023], [724.3154581696116, 353.32792805447053]]}]]]	2026-04-15 17:51:20.000 +0200	[{"tip": [418, 594], "nock": null, "score": 10}, {"tip": [430, 596], "nock": null, "score": "X"}, {"tip": [435, 510], "nock": null, "score": 8}]
 * New format rawBoundary: TargetBoundary[], rawRings: RingSet[]
 * → TargetData[]
*/
export function annotationToTargets(rawBoundary: any, rawRings: any): TargetData[] {
  const result = [] as TargetData[];
  const boundaryCentroids = [];

  if (rawBoundary == null || rawRings == null || !isBoundaryRingsArrayPair(rawBoundary, rawRings)) {
    logEvent('error', 'missing_boundary_or_rings', 'null', 'cannot convert to TargetData: missing or non-array boundary/rings');
    return result;
  }

  if (isOldAnnotationBoundary(rawBoundary)) {
    logEvent('warn', 'old_annotation_boundary', 'null', 'migrating old annotation boundary format to new multi-target format');
    boundaryCentroids.push(splineCentroid({ points: rawBoundary as [number, number][] }));
    result.push({
      paperBoundary: { points: rawBoundary as [number, number][] },
      ringSets: [] as RingSet[],
    });
  } else {
    for (const boundary of rawBoundary as TargetBoundary[]) {
      if (boundary.points.length === 0 || boundary.points.every(p => p[0] === 0 && p[1] === 0)) {
        logEvent('warn', 'empty_boundary', 'null', 'skipping empty boundary during migration');
        continue;
      }
      boundaryCentroids.push(splineCentroid(boundary));

      result.push({
        paperBoundary: boundary as TargetBoundary,
        ringSets: [] as RingSet[],
      });
    }
  }

  if (result.length === 0) {
    logEvent('error', 'no_valid_targets', 'null', 'no valid targets found during migration');
    return result;
  }

  if (isOldAnnotationRings(rawRings)) {
    logEvent('warn', 'old_annotation_rings', 'null', 'migrating old annotation rings format to new multi-target format');
    result[0].ringSets = [rawRings as RingSet];
  } else {
    for (const rings of rawRings as RingSet[]) {
      let closest_target_idx = 0;
      if (rings.length === 0 || rings[0].points.length === 0) {
        // skip empty rings
        logEvent('warn', 'empty_annotation_ring', 'null', 'skipping empty ring set during annotation migration');
        continue;
      }
      const centroid = splineCentroid(rings[0]);
      let max_dist = Infinity;
      for (let i = 0; i < boundaryCentroids.length; i++) {
        // find the closest target centroid for this ring set
        const dist = Math.hypot(centroid[0] - boundaryCentroids[i][0], centroid[1] - boundaryCentroids[i][1]);
        if (dist < max_dist) {
          max_dist = dist;
          closest_target_idx = i;
        }  
      }
      result[closest_target_idx].ringSets.push(rings);
    }
  }

  return result;
}