/**
 * Browser-only transport for the deployed research dashboard.
 * Model predictions are computed from fitted weights and the available synthetic
 * history on each request. No real patient trajectories or precomputed scores
 * enter this runtime. The local FastAPI dashboard does not load this module.
 */
import { predict } from './inference.js';

const FAMILIES = ['logistic', 'xgboost', 'gru'];
const DYNAMIC = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets'];
const LEVELS = {missingness:{low:.1,medium:.3,high:.5},delay:{low:1,medium:3,high:6},noise:{low:.1,medium:.25,high:.5}};
const OBSERVATIONS = {HR:['Heart rate','bpm'],O2Sat:['Oxygen saturation','%'],Temp:['Temperature','°C'],SBP:['Systolic blood pressure','mmHg'],MAP:['Mean arterial pressure','mmHg'],Resp:['Respiratory rate','/min'],Lactate:['Lactate','raw units'],WBC:['White blood cells','10³/µL'],Creatinine:['Creatinine','mg/dL'],Platelets:['Platelets','10³/µL']};
const observed = value => typeof value === 'number' && Number.isFinite(value);
function failure(status, message) { const error = new Error(message); error.status = status; return error; }

/** Seeded Mulberry32 with Box–Muller Gaussian draws, one draw per raw cell.
 * The distribution/severity matches the Python experiment definition; individual
 * random masks differ from NumPy PCG64. Replayed prefixes remain identical.
 */
function generator(seed) {
  let value = seed >>> 0;
  return () => { value += 0x6D2B79F5; let t = value; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function perturbRows(rows, kind='none', severity='medium', scales=null, seed=2019) {
  const result = rows.map(row => ({...row}));
  if (kind === 'none') return result;
  if (!LEVELS[kind] || !(severity in LEVELS[kind])) throw failure(422, 'Unsupported data perturbation or severity');
  const level = LEVELS[kind][severity], random = generator(seed);
  if (kind === 'noise' && (!scales || DYNAMIC.some(key => !observed(scales[key]) || scales[key]<0))) throw failure(409, 'Noise replay requires frozen training-only measurement scales');
  for (let i=0;i<result.length;i++) {
    for (const key of DYNAMIC) {
      const value = rows[i][key];
      if (kind === 'delay') result[i][key] = i>=level ? (rows[i-level][key] ?? null) : null;
      else if (kind === 'missingness') { const remove = random()<level; result[i][key] = observed(value) && !remove ? value : null; }
      else {
        const normal = Math.sqrt(-2*Math.log(Math.max(random(),Number.MIN_VALUE)))*Math.cos(2*Math.PI*random());
        result[i][key] = observed(value) ? value + normal*level*scales[key] : null;
      }
    }
  }
  return result;
}
function measurements(rows) {
  const current = rows.at(-1).hour;
  return Object.entries(OBSERVATIONS).map(([feature,[label,unit]]) => {
    const last = rows.findLast(row=>observed(row[feature]));
    if (!last) return {feature,label,unit,value:null,age_hours:null,status:'unmeasured'};
    const age = current-last.hour;
    return {feature,label,unit,value:Math.round(last[feature]*100)/100,age_hours:age,status:age===0?'observed':age>=4?'stale':'carried'};
  });
}
function validateReplay(input) {
  if (!input || typeof input!=='object' || Array.isArray(input)) throw failure(422, 'Replay requires a JSON object');
  const permitted = ['patient_id','family','hour','stress','severity'];
  if (Object.keys(input).some(key=>!permitted.includes(key))) throw failure(422, 'Unknown replay field');
  const result = {family:'logistic',hour:12,stress:'none',severity:'medium',...input};
  if (typeof result.patient_id!=='string' || !/^[A-Za-z0-9_-]{1,64}$/.test(result.patient_id)) throw failure(422, 'Invalid synthetic patient identifier');
  if (!FAMILIES.includes(result.family)) throw failure(422, 'Unsupported model family');
  if (!Number.isInteger(result.hour) || result.hour<0 || result.hour>500) throw failure(422, 'Replay hour must be an integer between 0 and 500');
  if (!['none',...Object.keys(LEVELS)].includes(result.stress) || !['low','medium','high'].includes(result.severity)) throw failure(422, 'Unsupported data perturbation or severity');
  return result;
}

export function createSiteRuntime({baseUrl='/data',loadJson}={}) {
  const prefix = baseUrl.replace(/\/$/,'');
  const getJson = loadJson || (async name => {
    const response = await fetch(`${prefix}/${name}`);
    if (!response.ok) throw failure(response.status, 'A published research asset could not be loaded');
    return response.json();
  });
  let pending;
  function load() {
    if (!pending) pending = Promise.all(['report.json','models.json','synthetic-patients.json','training-noise-scales.json'].map(getJson)).then(([report,models,cohort,scales])=>{
      if (!report.metadata || !report.models || cohort.source!=='synthetic' || !Array.isArray(cohort.patients)) throw failure(503, 'Invalid research dataset manifest');
      for (const patient of cohort.patients) {
        if (!/^([AB])_synthetic_\d+$/.test(patient.id) || !Array.isArray(patient.rows) || patient.rows.length!==patient.hours) throw failure(503, 'Replay requires generated synthetic trajectories');
        if (patient.rows.some((row,i)=>row.patient_id!==patient.id || row.hour!==i || row.hospital!==patient.hospital || Object.hasOwn(row,'SepsisLabel'))) throw failure(503, 'Invalid synthetic replay trajectory');
      }
      const available=FAMILIES.filter(family=>Object.hasOwn(models,family) && Object.hasOwn(report.models,family));
      if (!available.length) throw failure(503, 'No fitted models are available in this published run');
      return {report,models,cohort,scales,available,source:report.metadata.source};
    }).catch(error=>{pending=undefined;throw error;});
    return pending;
  }
  async function request(url,options={}) {
    const method=(options.method||'GET').toUpperCase();
    if (!['/api/status','/api/patients','/api/report','/api/replay'].includes(url)) throw failure(404,'Unknown research endpoint');
    if ((url==='/api/replay' && method!=='POST') || (url!=='/api/replay' && method!=='GET')) throw failure(405,'Unsupported request method');
    const item=await load();
    if (url==='/api/status') return {training_source:item.source,replay_source:'synthetic',available_models:item.available,demo:item.source==='synthetic',runtime:'browser',intended_use:'Retrospective research and software demonstration only',patient_uploads:false,clinical_use:false,perturbation_rng:'Mulberry32/Box–Muller; deterministic browser realizations, distinct from NumPy benchmark masks'};
    if (url==='/api/patients') return {source:'synthetic',patients:item.cohort.patients.map(({id,hospital,hours})=>({id,hospital,hours}))};
    if (url==='/api/report') return {training_source:item.source,demo:item.source==='synthetic',report:item.report};
    let body;
    try { body=JSON.parse(options.body); } catch (_) { throw failure(422,'Replay requires valid JSON'); }
    const input=validateReplay(body);
    if (!item.available.includes(input.family)) throw failure(404,'This model has not been trained in the loaded run');
    const patient=item.cohort.patients.find(candidate=>candidate.id===input.patient_id);
    if (!patient) throw failure(404,'Synthetic patient not found');
    if (input.hour>=patient.hours) throw failure(422,'Requested hour exceeds this synthetic timeline');
    const rows=patient.rows.slice(0,input.hour+1), bundle=item.models[input.family];
    const original=predict(bundle,rows);
    const changed=perturbRows(rows,input.stress,input.severity,item.scales,2019);
    const output=input.stress==='none'?original:predict(bundle,changed);
    return {patient_id:input.patient_id,source:'synthetic',training_source:item.source,
      out_of_distribution_illustration:item.source!=='synthetic',family:input.family,hour:input.hour,
      threshold:bundle.threshold,stress:input.stress,severity:input.severity,
      hours:rows.map(row=>row.hour),baseline_scores:original.scores,scores:output.scores,
      measurements:measurements(changed),contributions:output.contributions||[],
      contribution_method:output.contribution_method||null,contribution_scale:'uncalibrated_log_odds',
      contribution_unavailable_reason:output.contribution_unavailable_reason||null,
      perturbation_rng:'Mulberry32/Box–Muller; deterministic browser realization',
      warning:'Synthetic illustration, not a clinical assessment. Contributions explain the uncalibrated score, not causes.'};
  }
  return {request};
}

// A static export explicitly opts in. The FastAPI index never loads this file.
if (typeof document!=='undefined' && document.documentElement.dataset.runtime==='static') {
  globalThis.clinicalLabSite=createSiteRuntime({baseUrl:document.documentElement.dataset.base||'/data'});
  await import('./app.js');
}
