/* Actual fitted-model inference, no API keys, network calls, or clinical labels.
 * JSON is exported by clinical_lab.export_models. Floating-point neural kernels
 * may differ from PyTorch by a few float32 ULPs; parity is regression-tested.
 */
export const VARIABLES = Object.freeze([
 'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN',
 'Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium',
 'Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'
]);
export const DYNAMIC_COLUMNS = Object.freeze(VARIABLES.slice(0,34));
const FORMAT = 'clinical-lab-portable-v1';
const f32 = Math.fround;
const present = value => typeof value === 'number' && Number.isFinite(value);
const sigmoid = value => value >= 0 ? 1/(1+Math.exp(-value)) : Math.exp(value)/(1+Math.exp(value));
function dot(a,b) { let value=0; for(let i=0;i<a.length;i++)value+=a[i]*b[i]; return value; }
function checkRows(rows) {
 if(!Array.isArray(rows)||!rows.length)throw new Error('Inference needs at least one canonical patient-hour');
 const lastHours=new Map();
 for(const row of rows) {
  if(typeof row.patient_id!=='string'||!row.patient_id||!Number.isInteger(row.hour)||row.hour<0)throw new Error('Invalid patient/time metadata');
  if(lastHours.has(row.patient_id)&&row.hour!==lastHours.get(row.patient_id)+1)throw new Error('Patient rows must be chronological and consecutive');
  lastHours.set(row.patient_id,row.hour);
  for(const name of VARIABLES)if(!(name in row)||(row[name]!==null&&!present(row[name])))throw new Error(`Invalid or absent measurement field: ${name}`);
 }
}

/** Construct exactly the causal Python feature definitions, ignoring labels. */
export function buildFeatures(rows,ablation='full') {
 if(!['full','no_trends','no_missingness','values_only'].includes(ablation))throw new Error('Unsupported feature ablation');
 checkRows(rows);
 const patients=new Map(),output=[];
 for(const row of rows) {
  if(!patients.has(row.patient_id))patients.set(row.patient_id,{last:Array(40).fill(null),lastHour:Array(34).fill(null),raw:[],carried:[]});
  const state=patients.get(row.patient_id),raw=VARIABLES.map(name=>row[name]),features={};
  for(let j=0;j<40;j++) {
   if(present(raw[j])) {state.last[j]=raw[j];if(j<34)state.lastHour[j]=row.hour;}
   features[`${VARIABLES[j]}_value`]=present(state.last[j])?f32(state.last[j]):null;
  }
  const window=[...state.raw,raw].slice(-6);
  if(ablation==='full'||ablation==='no_missingness') {
   for(let j=0;j<34;j++) {
    const observed=window.map(values=>values[j]).filter(present);
    features[`${VARIABLES[j]}_mean6`]=observed.length?f32(observed.reduce((a,b)=>a+b,0)/observed.length):null;
    const previous=state.carried.length>=5?state.carried[state.carried.length-5][j]:null;
    features[`${VARIABLES[j]}_slope6`]=present(previous)&&present(state.last[j])?f32((state.last[j]-previous)/5):null;
   }
  }
  if(ablation==='full'||ablation==='no_trends')for(let j=0;j<34;j++) {
   features[`${VARIABLES[j]}_missing`]=present(raw[j])?0:1;
   features[`${VARIABLES[j]}_since`]=state.lastHour[j]===null?48:Math.min(48,Math.max(0,row.hour-state.lastHour[j]));
  }
  state.raw=window;
  state.carried.push([...state.last]); if(state.carried.length>5)state.carried.shift();
  output.push(features);
 }
 return output;
}

/** Apply exported, training-only imputation/scaling, retaining float32 casts. */
export function preprocess(bundle,features) {
 const p=bundle.preprocessing,columns=bundle.feature_columns;
 if(p.medians.length!==columns.length||(p.means&&p.means.length!==columns.length)||(p.scales&&p.scales.length!==columns.length))throw new Error('Malformed preprocessing dimensions');
 return features.map(row=>columns.map((column,j)=>{
  if(!(column in row))throw new Error(`Missing feature ${column}`);
  let value=present(row[column])?row[column]:p.medians[j];
  if(p.means)value-=p.means[j];
  if(p.scales)value/=p.scales[j];
  if(!Number.isFinite(value))throw new Error('Nonfinite preprocessed feature');
  return f32(value);
 }));
}
function treePrediction(tree,values) {
 let node=0;
 while(tree.left[node]!==-1) {
  const value=values[tree.features[node]];
  const left=!present(value)?Boolean(tree.default_left[node]):value<f32(tree.values[node]);
  node=left?tree.left[node]:tree.right[node];
 }
 return f32(tree.values[node]);
}
function gruStep(model,input,previous) {
 const size=model.hidden_size,ih=[],hh=[];
 for(let j=0;j<3*size;j++) {
  // Separate input and recurrent affine transforms follow torch.nn.GRU,
  // including applying reset *after* the candidate recurrent affine.
  ih.push(f32(dot(model.weight_ih[j],input)+model.bias_ih[j]));
  hh.push(f32(dot(model.weight_hh[j],previous)+model.bias_hh[j]));
 }
 const hidden=[];
 for(let j=0;j<size;j++) {
  const reset=f32(sigmoid(f32(ih[j]+hh[j])));
  const update=f32(sigmoid(f32(ih[size+j]+hh[size+j])));
  const candidate=f32(Math.tanh(f32(ih[2*size+j]+f32(reset*hh[2*size+j]))));
  // PyTorch's fused CPU cell computes candidate + update*(previous-candidate).
  hidden.push(f32(candidate+f32(update*f32(previous[j]-candidate))));
 }
 const logit=f32(dot(model.output_weight,hidden)+model.output_bias);
 return {hidden,probability:f32(sigmoid(logit))};
}
function calibrate(bundle,probability) {
 const c=bundle.calibration;
 if(!c)return probability;
 const p=Math.min(1-c.clip,Math.max(c.clip,probability));
 return sigmoid(c.coefficient*(Math.log(p)-Math.log1p(-p))+c.intercept);
}

/** Predict from canonical synthetic history using the real fitted parameters.
 * Attribution is exact only for logistic regression; no TreeSHAP surrogate is
 * invented in JavaScript. The local Python API retains native TreeSHAP support.
 */
export function predict(bundle,rows) {
 if(bundle.format!==FORMAT||bundle.family!==bundle.model.kind)throw new Error('Unsupported portable model format');
 const features=buildFeatures(rows,bundle.ablation),values=preprocess(bundle,features),model=bundle.model,states=new Map(),raw=[];
 for(let i=0;i<values.length;i++) {
  const x=values[i];
  if(model.kind==='logistic')raw.push(sigmoid(dot(model.coefficients,x)+model.intercept));
  else if(model.kind==='xgboost') {
   const base=f32(model.base_score);
   let margin=f32(-Math.log(f32(1/base-1)));
   for(const tree of model.trees)margin=f32(margin+treePrediction(tree,x));
   raw.push(f32(sigmoid(margin)));
  } else if(model.kind==='gru') {
   const id=rows[i].patient_id,previous=states.get(id)||Array(model.hidden_size).fill(0);
   const next=gruStep(model,x,previous);states.set(id,next.hidden);raw.push(next.probability);
  } else throw new Error('Unsupported model family');
 }
 let contributions=[],method=null,unavailable=null,bias=null;
 if(model.kind==='logistic') {
  contributions=bundle.feature_columns.map((feature,j)=>({feature,value:values.at(-1)[j]*model.coefficients[j]})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,8);
  method='coefficient_times_feature';bias=model.intercept;
 } else unavailable=model.kind==='xgboost'?'Native TreeSHAP is available in the local Python application; portable replay does not approximate it.':'No validated GRU feature attribution is available.';
 return {scores:raw.map(p=>calibrate(bundle,p)),raw_scores:raw,threshold:bundle.threshold,
         contributions,contribution_method:method,contribution_scale:'uncalibrated_log_odds',
         contribution_bias:bias,contribution_unavailable_reason:unavailable};
}
