import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import * as crypto from 'crypto';
import { Pool } from 'pg';
import { loadImageNode } from '../src/imageLoader';
import { findTarget, findRingSetFromCenter, ArcheryResult, TargetBoundary, isTargetBoundary } from '../src/targetDetection';
import { isSplineRing, splineCentroid, SplineRing, RingSet } from '../src/spline';
import { detectArrowsNN } from '../src/arrowDetector';

const IMAGES_DIR = path.resolve(__dirname, '../images');
const PORT = parseInt(process.env.ANNOTATE_PORT || '3737', 10);

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});
const K_POINTS = 8;

const RING_COLORS = [
  '#FFD700', '#FFD700',
  '#E8000D', '#E8000D',
  '#006CB7', '#006CB7',
  '#888888', '#888888',
  '#FFFFFF', '#FFFFFF',
];

interface TargetData {
  paperBoundary: TargetBoundary;
  ringSets: RingSet[];
}

export function isTargetData(x: any): x is TargetData {
  return typeof x === "object" && x != null && 
    isTargetBoundary((x as TargetData).paperBoundary) &&
    Array.isArray((x as TargetData).ringSets) &&
    (x as TargetData).ringSets.every(rs => rs.every(r => isSplineRing(r)));
}

interface ArrowData {
  tip: [number, number]
  score: number | 'X'
}

interface ImageData {
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


// ---------------------------------------------------------------------------
// Logging (AW-6)
// ---------------------------------------------------------------------------

const LOG_PATH = path.resolve(__dirname, '../logs/annotate.log');
fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

function logEvent(level: 'info' | 'warn' | 'error', event: string, filename: string, detail = '') {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, filename, detail });
  fs.appendFileSync(LOG_PATH, line + '\n');
  if (level === 'warn')  console.warn(`[warn]  ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
  if (level === 'error') console.error(`[error] ${event}: ${filename}${detail ? ' — ' + detail : ''}`);
}

// ---------------------------------------------------------------------------
// Validity check (AW-2)
// ---------------------------------------------------------------------------

/** True if a boundary polygon is non-degenerate (non-empty, non-null, not all-zero). */
export function isBoundaryValid(pts: TargetBoundary): boolean {
  return pts.points.length >= 3 &&
    pts.points.every(p => p[0] != null && p[1] != null) &&
    pts.points.some(p => p[0] !== 0 || p[1] !== 0);                   // at least one non-zero vertex
}

// ---------------------------------------------------------------------------
// Worker process for background detection (AW-5)
// ---------------------------------------------------------------------------

function clampBoundary(
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

function isOldFormatBoundary(pts: any): boolean {
  return Array.isArray(pts) && pts.length > 0 &&
  Array.isArray(pts[0]) && pts[0].length === 2 &&
  typeof pts[0]?.[0] === 'number' && typeof pts[0]?.[1] === 'number';
}

function isOldFormatRings(rings: any): boolean {
  return Array.isArray(rings) && rings.length > 0 &&
    rings.every(r => isSplineRing(r));
}

/** Convert DB format ex: 
 * old format: [[[529, 230], [574, 276], [549, 638], [496, 739], [97, 745], [51, 701], [8, 320], [49, 242]]]	[[[{"points": [[317.78895751251116, 512.155116841257], [313.2275451647999, 527.5573404446418], [300.123790444337, 537.0707728514956], [284.01566896411026, 536.8151611255315], [271.30574465375895, 527.2108641597003], [267.57416425538173, 512.155116841257], [273.7007596941402, 498.83944980486314], [285.16495409210086, 491.0322084749196], [299.2085657495854, 490.0562328068718], [312.0251603030828, 497.6264769749396]]}, {"points": [[343.4622351675308, 512.155116841257], [334.80010697590683, 543.2307240384354], [308.3636469637962, 562.4304436101214], [275.6097547201046, 562.6859050092147], [250.8626462098797, 542.0636445833788], [243.80809239643702, 512.155116841257], [255.50650877719806, 485.62055274850417], [278.2758157128545, 469.82962069979106], [306.41735850346464, 467.8698500253494], [332.0573129154203, 483.07226617458264]]}, {"points": [[368.01655103515196, 512.155116841257], [353.7507852142809, 556.9991977131598], [315.37753452667476, 584.0169698939909], [268.7110826706087, 583.9178344043194], [230.92164030010406, 556.5516334280371], [217.9508427639218, 512.155116841257], [234.1786922080015, 470.1249869814855], [270.3555157136608, 445.4534437827833], [314.5179703891795, 442.93873018363854], [352.64781358522436, 468.11239176504716]]}, {"points": [[392.7550231611233, 512.155116841257], [372.9984726571922, 570.9834612061896], [322.6102563844269, 606.2769988845623], [261.42184759531756, 606.3517931941432], [211.23626192351963, 570.8538979985028], [192.70627590024347, 512.155116841257], [213.12229165744887, 454.8266164947935], [262.34188857760074, 420.7900354730704], [322.5586548122715, 418.1920481070666], [373.485189778703, 452.9731517884385]]}, {"points": [[417.87494517256766, 512.155116841257], [391.9375730874133, 584.7435231109099], [329.82782622802193, 628.4903947706073], [253.8305804511231, 629.71531111013], [190.63459321890113, 585.8218864602853], [166.03402529923426, 512.155116841257], [190.66609996544088, 438.51123821350905], [253.54285744858882, 393.7094022242179], [331.1231166293886, 391.83334496775933], [395.01477745170615, 437.33099073358164]]}, {"points": [[442.9997196252475, 512.155116841257], [411.4913948231174, 598.9502061869346], [337.2186178660157, 651.2369125215531], [246.14940332770271, 653.3555434890079], [170.04935471894544, 600.7779376796364], [138.9035129853115, 512.155116841257], [168.28690045853293, 422.2517980290238], [244.33444419144578, 365.36882033920244], [339.69526451936235, 365.4509865285555], [417.117844739088, 421.27217234993464]]}, {"points": [[465.9935160440978, 512.155116841257], [428.2533738348362, 611.1284967924815], [341.67909545238683, 664.9648509570669], [239.20240282905635, 674.7362125564401], [150.24253394607712, 615.1684353157052], [113.32127586492405, 512.155116841257], [146.54717650802337, 406.4569640318819], [236.2543379964084, 340.50081052410815], [347.9379333028438, 340.08266051124633], [437.07506113155375, 406.77240590020256]]}, {"points": [[490.55371227533317, 512.155116841257], [447.65000977480463, 625.2209777031059], [350.4190781655502, 691.863751868566], [232.0019260029063, 696.8970015440941], [130.49703524511878, 629.5143798586261], [87.09732455906067, 512.155116841257], [124.39088659615368, 390.3594771480924], [227.77261254841451, 314.3967437459769], [356.42515013022233, 313.9616930051866], [458.5527131196098, 391.167978329181]]}, {"points": [[513.6521046182997, 512.155116841257], [464.27219703082864, 637.297703653076], [355.1989255836233, 706.5746095773792], [224.86016524146865, 718.877081066015], [110.53774253881261, 644.0156548386648], [62.09107710372453, 512.155116841257], [103.11366376404666, 374.90066988271997], [219.60512342818518, 289.2597969405891], [364.54977524490545, 288.9566680440057], [478.6334765440133, 376.5784497065372]]}, {"points": [[540.0777640321483, 512.155116841257], [485.49417163351654, 652.7163707301784], [364.1830420859712, 734.2248770327199], [217.9856999396448, 740.0345097523207], [90.03775517118848, 658.9097674848164], [36.08045504537941, 512.155116841257], [80.83439544720827, 358.71383395769453], [211.10202163642955, 263.08994054117744], [373.03394877003615, 262.8450668591728], [501.1958079703438, 360.18595639435625]]}]]]	[]	2026-04-15 11:40:00.167 +0200	583	1200
 * new format: rawBoundary: TargetBoundary[], rawRings: RingSet[]
 * → TargetData[] 
*/
function generateddbToTargets(rawBoundary: any, rawRings: any): TargetData[] {
  let result = [] as TargetData[];
  let targetsCentroid = [];

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
      let target = {
        paperBoundary: rawBoundary as TargetBoundary,
        ringSets: [] as RingSet[],
      } as TargetData;
      result.push(target);
      targetsCentroid.push(splineCentroid(target.paperBoundary));
    }
  }

  if (isOldFormatRings(rawRings)) {
    logEvent('warn', 'old_format_rings', 'null', 'migrating old rings format to new multi-target format');
    result[0].ringSets = [rawRings as RingSet];
  } else {
    for (const rings of rawRings as RingSet[]) {
        const centroid = splineCentroid(rings[0]);
        let max_dist = Infinity;
        let closest_target_idx = 0;
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

function isOldAnnotationBoundary(pts: any): boolean {
  return Array.isArray(pts) && pts.length > 0 &&
    Array.isArray(pts[0]) && pts[0].length === 2 &&
    typeof pts[0]?.[0] === 'number' && typeof pts[0]?.[1] === 'number';
}

function isOldAnnotationRings(rings: any): boolean {
  return isSplineRing(rings);
}

/* Convert DB format ex:
 * old format rawBoundary: [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]	rawRings: [[[{"points": [[457.30871071476906, 571.6529817201078], [453.86854999412947, 593.4868550286966], [435.31679070577894, 607.0463091200133], [412.49682012154534, 606.4923021061218], [394.34928520452786, 593.0623855167768], [388.5098006123553, 571.6529817201078], [398.1278187510277, 552.988843238466], [415.06058755293225, 544.7041261508197], [432.5846108956093, 544.6684391425641], [447.8855782083494, 554.1659918577445]]}, {"points": [[491.0746265153625, 571.6529817201078], [483.99711763454275, 615.3765407273429], [446.89561364582937, 642.6822618624738], [401.0783668957828, 641.6346876190569], [364.9654258251155, 614.411008992849], [353.665203087992, 571.6529817201078], [372.9460075007156, 534.6931864329105], [406.1793100437917, 517.370364571853], [441.40568310771897, 517.5199704151199], [472.1361990210538, 536.5468845067828]]}, {"points": [[528.1876967239184, 571.6529817201078], [512.1015213336848, 635.7955852390007], [456.6105730937149, 672.5818326195562], [391.6791295207504, 670.5625657501962], [338.57692011349127, 633.5833806428565], [318.14301267354597, 571.6529817201078], [341.7045239887297, 511.9949200234744], [394.4371492163758, 481.231709502451], [452.64663982009836, 482.92386299933014], [501.87857773707486, 514.937781485552]]}, {"points": [[565.4988880107312, 571.6529817201078], [540.3053120753408, 656.2868386637778], [466.27541230873766, 702.3271491610776], [382.29069583260724, 699.4571935520561], [312.04573121059906, 652.8594176993515], [282.18995596190814, 571.6529817201078], [311.50876229197144, 490.0564149852642], [382.4480108031212, 444.3329355830616], [464.02630061184334, 447.90086832193793], [531.7697933933273, 493.22054209750496]]}, {"points": [[603.1192344351948, 571.6529817201078], [567.8902162978543, 676.3284447123884], [475.6146685113878, 731.0704242254358], [373.40895647073995, 726.7923765675564], [287.4087619451176, 670.7592236318849], [247.31594008950717, 571.6529817201078], [279.7152435756911, 466.9570715229521], [370.2746488543679, 406.8671799213082], [476.32002592134825, 410.06467232631985], [563.8913231211919, 469.8828846856228]]}, {"points": [[641.3014576450207, 571.6529817201078], [595.920610973559, 696.6937185210629], [484.956323871505, 759.8210831372332], [364.6109571109527, 753.8698343572521], [262.1175913127149, 689.1343346793656], [210.48371649436643, 571.6529817201078], [248.24776107240675, 444.09460723505146], [357.97992847099346, 369.02792140322356], [488.786828365769, 371.6957996819099], [596.2566394414232, 446.3681059466288]]}, {"points": [[677.3820066499252, 571.6529817201078], [620.7628051606512, 714.742629086953], [493.9486253684132, 787.4965414155843], [355.3391946822307, 782.4053849447298], [237.7477701861023, 706.8400461277332], [175.24014133977445, 571.6529817201078], [217.02533311753146, 421.4101854982511], [345.1663183446101, 329.5916844654713], [501.61669898326954, 332.20951809834], [628.8965830897062, 422.65379877445287]]}, {"points": [[714.450074255263, 571.6529817201078], [646.0429550028542, 733.1097330616616], [503.50887490524457, 816.9199640263774], [345.98207988127297, 811.2036231230962], [213.58080585743105, 724.3983734853014], [140.1424713728702, 571.6529817201078], [185.0289001690232, 398.1634162166879], [332.21259607125694, 289.724226679632], [514.7115087051111, 291.9078377949859], [662.0836627169866, 398.5419740449332]]}, {"points": [[751.5317467638911, 571.6529817201078], [673.8357790109557, 753.3024016769157], [512.8158045972668, 845.563748321162], [336.5905467516034, 840.1077900251162], [186.91562729402486, 743.771759728473], [106.53204442679248, 571.6529817201078], [155.4759006045294, 376.6919052029593], [320.27283070904224, 252.97740738660855], [526.3876215263191, 255.97245758695294], [692.9781637407375, 376.0958051696731]]}, {"points": [[789.5736813681051, 571.6529817201078], [701.7904886946785, 773.6126871201835], [522.7671160635089, 876.1907357941182], [326.9831717952573, 869.6762497637326], [159.8655506462996, 763.4247907988499], [72.73764628574139, 571.6529817201078], [123.58796486957948, 353.52396376121624], [308.13390669270143, 215.61764078249513], [538.405578356426, 218.98498970045023], [724.3154581696116, 353.32792805447053]]}]]]	2026-04-15 17:51:20.000 +0200	[{"tip": [418, 594], "nock": null, "score": 10}, {"tip": [430, 596], "nock": null, "score": "X"}, {"tip": [435, 510], "nock": null, "score": 8}]
 * New format rawBoundary: TargetBoundary[], rawRings: RingSet[]
 * → TargetData[]
*/
function annotationToTargetData(rawBoundary: any, rawRings: any): TargetData[] {
  const result = [] as TargetData[];
  let needToUpdate = false;
  if (isOldAnnotationBoundary(rawBoundary)) {
    logEvent('warn', 'old_annotation_boundary', 'null', 'migrating old annotation boundary format to new multi-target format');
    result.push({
      paperBoundary: { points: rawBoundary as [number, number][] },
      ringSets: [] as RingSet[],
    });
    needToUpdate = true;
  } else {
    for (const boundary of rawBoundary as TargetBoundary[]) {
      result.push({
        paperBoundary: boundary as TargetBoundary,
        ringSets: [] as RingSet[],
      });
    }
  }

  if (isOldAnnotationRings(rawRings)) {
    logEvent('warn', 'old_annotation_rings', 'null', 'migrating old annotation rings format to new multi-target format');
    result[0].ringSets = [rawRings as RingSet];
    needToUpdate = true;
  } else {
    result[0].ringSets = rawRings as RingSet[];
  }

  return result;
}

async function loadImage(imgPath: string, filename: string): Promise<ImageData> {
  let base64: string;
  let width: number;
  let height: number;

  console.log(`fast path: loading ${filename}…`);
  ({ base64, width, height } = await loadImageBase64(imgPath));
  return {
    filename: filename,
    base64: base64,
    height: height,
    width: width,
    generated: {
      arrows: [],
      targets: [],
    },
    annotated: {
      arrows: [],
      targets: [],
    },
  }
}

async function fetchGeneratedData(data: ImageData): Promise<[ImageData, boolean]>  {
  const { rows } = await db.query(
    'SELECT rings, paper_boundary, arrows FROM generated WHERE filename = $1',
    [data.filename],
  );
  if (rows.length == 0) {
    return [data, false]
  }
  if (rows.length > 1) {
    console.warn(`multiple rows for image : ${data.filename}`)
  } 

  data.generated.targets = generateddbToTargets(rows[0].paper_boundary, rows[0].rings);
  data.generated.arrows = rows[0].arrows ?? [];
  console.log(`found generated data: ${JSON.stringify(data.generated, null, 2)}`)

  return [data, true];
}

async function fetchAnnotationData(data: ImageData): Promise<[ImageData, boolean]>  {
  const { rows } = await db.query(
    'SELECT rings, paper_boundary, arrows FROM annotation WHERE filename = $1',
    [data.filename],
  );
  if (rows.length == 0) {
    return [data, false]
  }
  if (rows.length > 1) {
    console.warn(`multiple rows for image : ${data.filename}`)
  } 

  data.annotated.targets = annotationToTargetData(rows[0].paper_boundary, rows[0].rings);
  data.annotated.arrows = rows[0].arrows ?? [];
  console.log(`found annotation data: ${JSON.stringify(data.annotated, null, 2)}`)

  return [data, true];
}


/** Convert TargetData[] → DB format { boundary: TargetBoundary[], rings: RingSet[], arrows: ArrowData[] } */
function targetsToDB(targets: TargetData[], arrows: ArrowData[]): { boundary: TargetBoundary[]; rings: RingSet[], arrows: ArrowData[] } {
  return {
    boundary: targets.map((t: TargetData) => t.paperBoundary ?? []),
    rings:    targets.map((t: TargetData) => t.ringSets ?? []).flat(),
    arrows:   arrows,
  };
}

/** Hash of algorithm source files — used to detect stale detections in DB. */
function computeAlgorithmHash(): string {
  const files = [
    path.resolve(__dirname, '../src/targetDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

async function loadImageBase64(imgPath: string): Promise<{ base64: string; width: number; height: number }> {
  const { Jimp } = require('jimp');
  const img = await Jimp.read(imgPath);
  img.scaleToFit({ w: 1200, h: 1200 });
  const base64 = await img.getBase64('image/jpeg');
  return { base64, width: img.width, height: img.height };
}

async function processImage(imgPath: string, imgData: ImageData): Promise<ImageData> {
  const filename = path.basename(imgPath);
  console.log(`  [1/3] loadImageNode ${filename}…`);
  const { rgba, width, height } = await loadImageNode(imgPath);
  console.log(`  [3/3] findTarget ${filename}…`);
  const result = findTarget(rgba, width, height);
  if (result.success) {
    console.log(`  findTarget success: ${filename}`);
    for (const t of result.targets) { 
      imgData.generated.targets.push({
        paperBoundary: t.paperBoundary,
        ringSets: [t.rings],
      });
    }
  }
  console.log(`  done: ${filename}`);
  const arrows = await detectArrowsNN(rgba, width, height, path.resolve(__dirname, '../models/arrow-detector.onnx'));
  for (const a of arrows) {
    imgData.generated.arrows.push({ tip: a.tip, score: a.score });
  }
  return imgData;
}


function generateHtml(filenames: string[]): string {
  const tpl = fs.readFileSync(path.join(__dirname, 'annotate.html'), 'utf-8');
  return tpl
    .replace('{{IMAGES}}',      JSON.stringify(filenames))
    .replace('{{RING_COLORS}}', JSON.stringify(RING_COLORS))
    .replace('{{K_POINTS}}',    String(K_POINTS));
}

type GenState = 'ready' | 'queued' | 'computing' | 'error' | 'unknown';

interface SSEMessage {
  type: 'status' | 'new_image' | 'removed';
  filename: string;
  state?: GenState; 
}

async function main(): Promise<void> {
  const jpgFiles = fs
    .readdirSync(IMAGES_DIR)
    .filter(f => /\.(jpg|jpeg)$/i.test(f))
    .map(f => path.join(IMAGES_DIR, f))
    .sort();

  if (jpgFiles.length === 0) {
    console.error(`No JPEG files found in ${IMAGES_DIR}`);
    process.exit(1);
  }

  const filenames = jpgFiles.map(f => path.basename(f));
  const currentHash = computeAlgorithmHash();

  // --- DB setup (AW-1) ---
  await db.query(`
    CREATE TABLE IF NOT EXISTS public.annotations (
      filename text NOT NULL,
      paper_boundary jsonb NULL,
      rings jsonb DEFAULT '[]'::jsonb NOT NULL,
      updated_at timestamptz DEFAULT now() NOT NULL,
      arrows jsonb DEFAULT '[]'::jsonb NOT NULL,
      CONSTRAINT annotations_pkey PRIMARY KEY (filename)
    )
  `);
  await db.query(`
    CREATE TABLE IF NOT EXISTS public."generated" (
      filename text NOT NULL,
      algorithm_hash text NOT NULL,
      paper_boundary jsonb NULL,
      rings jsonb DEFAULT '[]'::jsonb NOT NULL,
      arrows jsonb DEFAULT '[]'::jsonb NOT NULL,
      updated_at timestamptz DEFAULT now() NOT NULL,
      width int4 NULL,
      height int4 NULL,
      CONSTRAINT generated_pkey PRIMARY KEY (filename)
    )
  `);

  // Wipe annotations rows with corrupt ring data
  const { rowCount: wiped } = await db.query(`
    UPDATE annotations SET rings = '[]', paper_boundary = NULL
     WHERE rings::text LIKE '%null%' AND rings <> 'null'::jsonb
  `);
  if (wiped) {
    console.log(`Wiped ${wiped} annotation rows with corrupt ring data.`);
    logEvent('warn', 'db_wipe', '', `${wiped} rows with null coordinates reset`);
  }

  console.log('Tables ready.');
  console.log(`Algorithm hash: ${currentHash}`);

  const { rows: genRows } = await db.query('SELECT filename, algorithm_hash FROM generated');
  const inGenerated = new Map<string, string>(genRows.map((r: any) => [r.filename, r.algorithm_hash as string]));

  const { rows: annRows } = await db.query('SELECT filename FROM annotations');
  const inAnnotations = new Set<string>(annRows.map((r: any) => r.filename as string));

  const readyCount = [...inGenerated.values()].filter(h => h === currentHash).length;
  const staleCount = filenames.length - readyCount;
  console.log(`Images: ${filenames.length} total — ${readyCount} ready, ${staleCount} stale/new`);

  const generationStatus = new Map<string, GenState>(
    filenames.map(f => [f, inGenerated.get(f) === currentHash ? 'ready' : 'queued']),
  );
  const sseClients = new Set<http.ServerResponse>();
  function broadcastSSE(data: SSEMessage) {
    generationStatus.set(data.filename, data.state ?? 'unknown');
    const msg = `data: ${JSON.stringify(data)}\n\n`;
    for (const client of sseClients) {
      try { client.write(msg); } catch { sseClients.delete(client); }
    }
  }

  const imageCache = new Map<string, ImageData>();
  // Deduplicates concurrent /api/image/ requests for the same file so detection
  // runs at most once.  Second caller awaits the first caller's promise.

  const html = generateHtml(filenames);

  async function computeGeneratedData(filename: string, imgPath: string, imageData: ImageData): Promise<ImageData> {
    // Slow path: run detection synchronously
    broadcastSSE({ type: 'status', filename, state: 'computing' });
    imageData = await processImage(imgPath, imageData);
    const {boundary: dbBoundary, rings: dbRings, arrows: dbArrows} = targetsToDB(imageData.generated.targets, imageData.generated.arrows);
    await db.query(
      `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (filename) DO UPDATE
          SET algorithm_hash = EXCLUDED.algorithm_hash,
              paper_boundary = EXCLUDED.paper_boundary,
              rings          = EXCLUDED.rings,
              arrows         = EXCLUDED.arrows,
              width          = EXCLUDED.width,
              height         = EXCLUDED.height,
              updated_at     = NOW()`,
      [filename, currentHash,
        JSON.stringify(dbBoundary), JSON.stringify(dbRings),
        JSON.stringify(dbArrows), imageData.width, imageData.height],
    );
    inGenerated.set(filename, currentHash);
    if (!inAnnotations.has(filename)) {
      await db.query(`INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (filename) DO UPDATE
                 SET paper_boundary = EXCLUDED.paper_boundary,
                     rings          = EXCLUDED.rings,
                     arrows         = EXCLUDED.arrows,
                     updated_at     = NOW()`, 
        [filename, 
          JSON.stringify(dbBoundary), 
          JSON.stringify(dbRings), 
          JSON.stringify(dbArrows)],);
      inAnnotations.add(filename);
    }

    broadcastSSE({ type: 'status', filename, state: 'ready' });
    return imageData;
  }

  // --- HTTP server ---
  const server = http.createServer(async (req, res) => {
    const respond = (status: number, body: string, type = 'application/json') => {
      res.writeHead(status, { 'Content-Type': type });
      res.end(body);
    };

    if (req.method === 'GET' && req.url === '/') {
      respond(200, html, 'text/html; charset=utf-8');

    } else if (req.method === 'GET' && req.url === '/api/stale-images') {
      const stale = filenames.filter(f => inGenerated.get(f) !== currentHash);
      respond(200, JSON.stringify({ stale }));

    } else if (req.method === 'GET' && req.url === '/api/generation-status') {
      const out: Record<string, string> = {};
      for (const [f, s] of generationStatus) out[f] = s;
      respond(200, JSON.stringify(out));

    } else if (req.method === 'GET' && req.url === '/api/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      res.write(`data: ${JSON.stringify({ type: 'connected' })}\n\n`);
      sseClients.add(res);
      req.on('close', () => sseClients.delete(res));

    } else if (req.method === 'GET' && req.url?.startsWith('/api/image-status/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image-status/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      if (!inGenerated.has(filename)) { respond(200, '{"state":"new"}'); return; }
      if (inGenerated.get(filename) !== currentHash) { respond(200, '{"state":"stale"}'); return; }
      respond(200, '{"state":"ready"}');

    } else if (req.method === 'GET' && req.url?.startsWith('/api/image/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      try {
      // if the image is not in cache, load it (and run detection if needed) before responding
      if (!imageCache.has(filename)) {
        try {
          const imgPath = path.join(IMAGES_DIR, filename);
          if (!fs.existsSync(imgPath)) { respond(404, '{"error":"not found"}'); return; }
  
          const hashInGen = inGenerated.get(filename);
          let isReady: boolean = false;
  
          console.log(`image request: ${filename}  hashInGen=${hashInGen ?? 'null'}  isReady=${isReady}`);
  
          let imageData: ImageData = await loadImage(imgPath, filename)
          
          if (hashInGen === currentHash) {
            [imageData, isReady] = await fetchGeneratedData(imageData)
          }
  
          if (!isReady) {
            imageData = await computeGeneratedData(filename, imgPath, imageData)
          }

          [imageData, isReady] = await fetchAnnotationData(imageData);
          
          imageCache.set(filename, imageData)
        } catch (err) {
          console.error('Compute error:', err);
          logEvent('error', 'compute_failed', filename, String(err));
          broadcastSSE({ type: 'status', filename, state: 'error' });
          respond(500, '{"error":"internal error"}');
          return;
        }
      }

      respond(200, JSON.stringify(imageCache.get(filename)!));
    } catch (e) {
      console.error('Error processing image:', e);
      respond(500, '{"error":"internal error"}');
    } 
    } else if (req.method === 'POST' && req.url?.startsWith('/api/recompute/')) {
      const filename = decodeURIComponent(req.url.slice('/api/recompute/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      imageCache.delete(filename);
      inGenerated.delete(filename);
      inAnnotations.delete(filename);
      await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
      await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
      broadcastSSE({ type: 'status', filename, state: 'computing' });
      respond(202, '{"status":"computing"}');
      try {
        const imgPath = path.join(IMAGES_DIR, filename);
        const imageData = await computeGeneratedData(filename, imgPath, await loadImage(imgPath, filename));
        imageCache.set(filename, imageData)
      } catch (err) {
        console.error('Recompute error:', err);
        logEvent('error', 'recompute_failed', filename, String(err));
        broadcastSSE({ type: 'status', filename, state: 'error' });
      }

    } else if (req.method === 'POST' && req.url === '/api/save') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      req.on('error', (err) => {
        console.error('Request error:', err);
        logEvent('error', 'save-request-error', '', String(err));
        respond(500, JSON.stringify({ error: String(err) }));
      });
      req.on('end', async () => {
        try {
          const data = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          const received = Object.keys(data).length;
          console.log(`[save] received ${received} annotation(s)`);
          let saved = 0;
          for (const [filename, ann] of Object.entries(data) as [string, any][]) {
            const imgMeta = imageCache.get(filename);
            const w = imgMeta?.width ?? 0, h = imgMeta?.height ?? 0;
            const targets = (ann.targets ?? []).map((t: any) => ({
              paperBoundary: clampBoundary(t.paperBoundary ?? [], w, h) ?? [],
              ringSets:      t.ringSets ?? [],
            }));
            const { boundary: dbBoundary, rings: dbRings, arrows: dbArrows } = targetsToDB(targets, ann.arrows ?? []);
            await db.query(
              `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (filename) DO UPDATE
                 SET paper_boundary = EXCLUDED.paper_boundary,
                     rings          = EXCLUDED.rings,
                     arrows         = EXCLUDED.arrows,
                     updated_at     = NOW()`,
              [filename, JSON.stringify(dbBoundary), JSON.stringify(dbRings), JSON.stringify(dbArrows)],
            );
            console.log(`[save]   OK ${filename}: targets=${targets.length}, arrows=${dbArrows.length}`);
            logEvent('info', 'save-ok', filename, `targets=${targets.length} arrows=${dbArrows.length}`);
            saved++;
          }
          console.log(`[save] done: saved=${saved}`);
          respond(200, JSON.stringify({ ok: true, saved }));
        } catch (e) {
          console.error('Save error:', e);
          logEvent('error', 'save-error', '', String(e));
          respond(500, JSON.stringify({ error: String(e) }));
        }
      });

    } else if (req.method === 'POST' && req.url === '/api/detect-ringset') {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
      req.on('error', (err) => {
        console.error('Request error:', err);
        logEvent('error', 'detect-ringset-request-error', '', String(err));
        respond(500, JSON.stringify({ error: String(err) }));
      });
      req.on('end', async () => {
        try {
          const { filename, cx, cy } = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
          const imgPath = path.join(IMAGES_DIR, filename);
          const { rgba, width, height } = await loadImageNode(imgPath);
          const t = findRingSetFromCenter(rgba, width, height, cx, cy);
          const boundary = clampBoundary(t.paperBoundary.points, width, height) ?? [];
          respond(200, JSON.stringify({ rings: t.rings, paperBoundary: boundary }));
        } catch (e) {
          console.error('detect-ringset error:', e);
          respond(500, JSON.stringify({ error: String(e) }));
        }
      });

    } else if (req.method === 'DELETE' && req.url?.startsWith('/api/image/')) {
      const filename = decodeURIComponent(req.url.slice('/api/image/'.length));
      if (!filenames.includes(filename)) { respond(404, '{"error":"not found"}'); return; }
      try {
        await db.query('DELETE FROM annotations WHERE filename = $1', [filename]);
        await db.query('DELETE FROM generated WHERE filename = $1', [filename]);
        const imgPath = path.join(IMAGES_DIR, filename);
        if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
        imageCache.delete(filename);
        inGenerated.delete(filename);
        inAnnotations.delete(filename);
        generationStatus.delete(filename);
        filenames.splice(filenames.indexOf(filename), 1);
        logEvent('info', 'delete', filename, 'image and all data removed');
        broadcastSSE({ type: 'removed', filename });
        respond(200, '{"ok":true}');
      } catch (err) {
        console.error('Delete error:', err);
        respond(500, JSON.stringify({ error: String(err) }));
      }

    } else {
      respond(404, '');
    }
  });

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Annotation tool: http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop.');
    if (!process.env.NO_BROWSER) require('child_process').exec(`open http://localhost:${PORT}`);

    fs.watch(IMAGES_DIR, (event, name) => {
      if (!name || !/\.(jpg|jpeg)$/i.test(name)) return;
      if (filenames.includes(name)) return;
      const fullPath = path.join(IMAGES_DIR, name);
      setTimeout(() => {
        if (!fs.existsSync(fullPath)) return;
        console.log(`New image detected: ${name}`);
        filenames.push(name);
        generationStatus.set(name, 'queued');
        broadcastSSE({ type: 'new_image', filename: name });
      }, 500);
    });
  });
}

main().catch(err => { console.error(err); process.exit(1); });
