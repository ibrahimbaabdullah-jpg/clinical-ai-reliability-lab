/* Browser-independent integration test of actual exported model weights.
 * Usage: node tests/test_site_runtime.cjs artifacts/browser-pilot/data
 * No browser/session automation or precomputed inference responses are used.
 */
'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict'),vm=require('node:vm');
const {pathToFileURL}=require('node:url');
const directory=path.resolve(process.argv[2]||'artifacts/browser-pilot/data');
const web=path.resolve(__dirname,'../clinical_lab/web');
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'clinical-site-runtime-'));
const read=name=>JSON.parse(fs.readFileSync(path.join(directory,name),'utf8'));
const input=Object.fromEntries(['report.json','models.json','synthetic-patients.json','training-noise-scales.json'].map(name=>[name,read(name)]));

(async()=>{
 // Use the exact browser module source; extension-only changes let all supported
 // Node versions load ES modules without requiring a repository package.json.
 fs.writeFileSync(path.join(temporary,'inference.mjs'),fs.readFileSync(path.join(web,'inference.js')));
 fs.writeFileSync(path.join(temporary,'site-adapter.mjs'),fs.readFileSync(path.join(web,'site-adapter.js'),'utf8').replace("'./inference.js'","'./inference.mjs'"));
 const {createSiteRuntime,perturbRows}=await import(pathToFileURL(path.join(temporary,'site-adapter.mjs')));
 let loads=0;
 const runtime=createSiteRuntime({loadJson:async name=>{loads++;return structuredClone(input[name]);}});
 const [status,listing,response]=await Promise.all(['/api/status','/api/patients','/api/report'].map(url=>runtime.request(url)));
 assert.equal(loads,4,'Concurrent page initialization must load every asset once');
 assert.equal(status.runtime,'browser');assert.equal(status.replay_source,'synthetic');
 assert.equal(status.training_source,input['report.json'].metadata.source);
 assert.deepEqual(response.report,input['report.json']);
 assert.deepEqual(status.available_models,['logistic','xgboost','gru']);
 assert.ok(listing.patients.length>0);
 assert.ok(listing.patients.every(p=>p.id.includes('_synthetic_')));
 const patient=listing.patients[0];
 const call=changes=>runtime.request('/api/replay',{method:'POST',body:JSON.stringify({patient_id:patient.id,family:'logistic',hour:12,stress:'none',severity:'medium',...changes})});
 let replayCount=0;
 for(const family of status.available_models)for(const stress of ['none','missingness','delay','noise']) {
   const a=await call({family,stress,hour:12}),b=await call({family,stress,hour:18});replayCount+=2;
   assert.equal(a.source,'synthetic');assert.equal(a.family,family);
   assert.equal(a.out_of_distribution_illustration,status.training_source!=='synthetic');
   assert.deepEqual(a.hours,Array.from({length:13},(_,i)=>i));
   assert.deepEqual(a.scores,b.scores.slice(0,13),`${family}/${stress} must be prefix-consistent`);
   assert.deepEqual(a.baseline_scores,b.baseline_scores.slice(0,13));
   assert.ok(a.scores.every(p=>Number.isFinite(p)&&p>=0&&p<=1));
   assert.equal(a.measurements.length,10);
   if(family==='logistic'){assert.equal(a.contributions.length,8);assert.ok(a.contributions.every(c=>Number.isFinite(c.value)));}
   else {assert.deepEqual(a.contributions,[]);assert.ok(a.contribution_unavailable_reason);}
 }
 for(const severity of ['low','medium','high'])for(const stress of ['missingness','delay','noise'])await call({severity,stress});
 for(const [changes,code] of [[{patient_id:'unknown'},404],[{patient_id:'../raw.parquet'},422],[{family:'invalid'},422],[{hour:-1},422],[{hour:500},422],[{hour:2.5},422],[{stress:'bad'},422],[{severity:'bad'},422],[{upload_path:'patient.csv'},422]]) {
   await assert.rejects(()=>call(changes),error=>error.status===code);
 }
 await assert.rejects(()=>runtime.request('/api/replay'),error=>error.status===405);
 await assert.rejects(()=>runtime.request('/api/patients',{method:'POST'}),error=>error.status===405);
 await assert.rejects(()=>runtime.request('/api/unknown'),error=>error.status===404);
 const rows=input['synthetic-patients.json'].patients[0].rows.slice(0,20),scales=input['training-noise-scales.json'];
 for(const stress of ['missingness','delay','noise']) {
   const before=JSON.stringify(rows),changed=perturbRows(rows,stress,'high',scales);
   assert.equal(JSON.stringify(rows),before,'Raw exported observations must remain immutable');
   assert.ok(changed.every((row,i)=>row.patient_id===rows[i].patient_id&&row.hour===rows[i].hour&&row.hospital===rows[i].hospital&&row.Age===rows[i].Age));
   assert.deepEqual(perturbRows(rows.slice(0,13),stress,'high',scales),changed.slice(0,13));
 }
 const unsafe=structuredClone(input);unsafe['synthetic-patients.json'].patients[0].rows[0].SepsisLabel=1;
 await assert.rejects(()=>createSiteRuntime({loadJson:async name=>unsafe[name]}).request('/api/status'),error=>error.status===503);
 const missingScales=structuredClone(input);missingScales['training-noise-scales.json']={};
 const restricted=createSiteRuntime({loadJson:async name=>missingScales[name]});
 await assert.rejects(()=>restricted.request('/api/replay',{method:'POST',body:JSON.stringify({patient_id:patient.id,stress:'noise'})}),error=>error.status===409);

 // Full UI routing against the actual static transport, with minimal DOM stubs.
 const html=fs.readFileSync(path.join(web,'index.html'),'utf8');
 const ids=new Set(Array.from(html.matchAll(/\bid="([^"]+)"/g),match=>match[1]));
 const elements=new Map();
 function element(id){assert.ok(ids.has(id),`Absent UI target ${id}`);if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',value:({['hour-slider']:'12',['stress-select']:'none',['severity-select']:'medium'})[id]||'',hidden:false,addEventListener(){},setAttribute(){},removeAttribute(){},classList:{toggle(){}},hasChildNodes(){return!!this.innerHTML;}});return elements.get(id);}
 let capturedBlob,clicked;
 const context=vm.createContext({document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>({click(){clicked=this;}})},location:{hash:'#overview'},window:{addEventListener(){}},clinicalLabSite:runtime,Blob,URL:{createObjectURL(blob){capturedBlob=blob;return 'blob:local-test';},revokeObjectURL(){}},setTimeout,clearTimeout,console});
 const app=fs.readFileSync(path.join(web,'app.js'),'utf8').replace(/init\(\);\s*$/,'globalThis.initialized=init();');
 vm.runInContext(app,context);await context.initialized;
 assert.equal(element('error').textContent,'');assert.equal(element('runtime-label').textContent,'INTERACTIVE RESEARCH EDITION');
 assert.ok(element('ablation-table').innerHTML.length);assert.ok(element('subgroup-table').innerHTML.length);
 for(const section of ['overview','replay','reliability','methods']){context.location.hash=`#${section}`;vm.runInContext('page()',context);}
 for(const family of status.available_models){element('report-model').value=family;vm.runInContext('reliability()',context);assert.ok(element('calibration-chart').innerHTML.includes('<svg'));}
 element('patient-select').value=patient.id;element('model-select').value='xgboost';
 await vm.runInContext('replay()',context);
 assert.ok(element('timeline-chart').innerHTML.includes('<svg'));
 assert.ok(element('contributions').innerHTML.length);
 assert.ok(!/\b(?:undefined|NaN)\b/.test([...elements.values()].map(e=>`${e.innerHTML} ${e.textContent}`).join(' ')));
 vm.runInContext("downloadReport('json')",context);
 assert.deepEqual(JSON.parse(await capturedBlob.text()),input['report.json']);assert.ok(clicked.download.endsWith('.json'));
 vm.runInContext("downloadReport('csv')",context);
 const csv=await capturedBlob.text();assert.equal(csv.split('\r\n').length,7);assert.ok(csv.includes('physionet-2019')||csv.includes('synthetic'));assert.ok(clicked.download.endsWith('.csv'));
 console.log(`Static runtime passed: ${replayCount} fitted-model/prefix replays, every stress severity, input guards, artifact provenance, all four UI pages, real JSON/CSV downloads.`);
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>fs.rmSync(temporary,{recursive:true,force:true}));
