'use strict';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = (value, digits=3) => finite(value) ? value.toFixed(digits) : '—';
const percent = value => finite(value) ? `${(100*value).toFixed(1)}%` : '—';
const names = {logistic:'Logistic regression',xgboost:'Gradient-boosted trees',gru:'Temporal GRU'};
const details = {logistic:'Regularized interpretable baseline',xgboost:'Nonlinear tabular learning',gru:'Causal patient trajectories'};
let status, report, patients = [], replaySequence = 0;
async function request(url, options) {
 if (globalThis.clinicalLabSite) return globalThis.clinicalLabSite.request(url, options);
 const response = await fetch(url, options);
 if (!response.ok) {
  let detail = `Request failed (${response.status})`;
  try { const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail; } catch (_) {}
  throw new Error(detail);
 }
 return response.json();
}
function page() {
 const requested = location.hash.slice(1), id = ['overview','replay','reliability','methods'].includes(requested) ? requested : 'overview';
 document.querySelectorAll('.page').forEach(section => section.hidden = section.id !== id);
 document.querySelectorAll('nav a').forEach(link => { const active = link.dataset.page === id; link.classList.toggle('active',active); if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current'); });
 $('breadcrumb').textContent = {overview:'Overview',replay:'Patient replay',reliability:'Reliability report',methods:'Methods & limitations'}[id];
 if (id === 'replay' && patients.length && !$('timeline-chart').hasChildNodes()) replay();
}
function metric(label,value,detail,definition='') { return `<div class="metric"${definition?' title="'+escapeHtml(definition)+'"':''}><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-detail">${escapeHtml(detail)}</div></div>`; }
function overview() {
 const metadata = report.metadata || {}, selected = report.selected_family || Object.keys(report.models)[0], evaluation = report.models[selected]?.external;
 $('overview-metrics').innerHTML = metric('Cohort',metadata.patients?.toLocaleString() || '—',metadata.source === 'synthetic' ? 'Generated synthetic patients' : 'Patients in the loaded experiment') + metric('Trained model families',Object.keys(report.models).length,'Available in this run') + metric('External event sensitivity',percent(evaluation?.pre_onset?.event_sensitivity),'Selected family · hospital B') + metric('False alerts / patient-day',number(evaluation?.pre_onset?.false_alerts_per_patient_day,2),'Eligible non-warning patient-days · B','Observed hours of never-septic patients plus hours before onset minus 6 in eligible septic patients, divided by 24. Excludes warning-window and post-onset hours and all censored septic cases.');
 $('model-table').innerHTML = Object.keys(names).map(family => {
  const evaluation = report.models[family]?.external, mark = family === selected ? '<span class="selected-mark">SELECTED</span>' : '';
  if (!evaluation) return `<tr><td><strong>${names[family]}</strong><span class="model-detail">${details[family]}</span></td><td colspan="5" class="na">Not trained in this run</td></tr>`;
  const h=evaluation.hourly_metrics,p=evaluation.pre_onset;
  return `<tr><td><strong>${names[family]}</strong>${mark}<span class="model-detail">${details[family]}</span></td><td>${number(h.average_precision)}</td><td>${number(h.brier)}</td><td>${percent(p.event_sensitivity)}</td><td>${number(p.false_alerts_per_patient_day,2)}</td><td>${finite(p.median_lead_time_hours)?number(p.median_lead_time_hours,1)+' h':'—'}</td></tr>`;
 }).join('');
 const synthetic = metadata.source === 'synthetic';
 $('evidence-tag').textContent = synthetic ? 'SYNTHETIC DEMONSTRATION' : metadata.complete_public_cohort ? 'RETROSPECTIVE RESEARCH' : 'REDUCED-COHORT PRELIMINARY RUN';
 $('evidence-text').textContent = synthetic ? 'All displayed results in this run use generated data. No clinical performance claims.' : 'Benchmark results use the loaded cohort. Patient replay remains a synthetic illustration, not clinical assessment.';
 $('run-heading').textContent = synthetic ? 'A synthetic software demonstration' : metadata.complete_public_cohort ? 'A complete public-cohort experiment' : 'A preliminary, reduced-cohort experiment';
 $('run-disclosure').textContent = report.disclosure || (synthetic ? 'The generated population has an enriched event rate. Results validate the workflow, not clinical utility or safety.' : 'Retrospective results describe this specific cohort and protocol. They do not establish prospective clinical benefit. See the run manifest for data provenance and cohort restrictions.');
}
function svgChart(series, maxHour, threshold) {
 const w=850,h=230,m={left:44,right:18,top:12,bottom:34},x=t=>m.left+t/Math.max(1,maxHour)*(w-m.left-m.right),y=v=>h-m.bottom-v*(h-m.top-m.bottom);
 const path = values => values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Calibrated model scores up to current replay hour">`;
 [0,.25,.5,.75,1].forEach(v=>svg+=`<line x1="${m.left}" y1="${y(v)}" x2="${w-m.right}" y2="${y(v)}" stroke="#e6ecee"/><text x="${m.left-10}" y="${y(v)+4}" text-anchor="end">${(v*100).toFixed(0)}%</text>`);
 [0,Math.round(maxHour/4),Math.round(maxHour/2),Math.round(maxHour*3/4),maxHour].filter((v,i,a)=>a.indexOf(v)===i).forEach(t=>svg+=`<text x="${x(t)}" y="${h-10}" text-anchor="middle">${t} h</text>`);
 if(threshold<=1)svg+=`<line x1="${m.left}" y1="${y(threshold)}" x2="${w-m.right}" y2="${y(threshold)}" stroke="#8b9fa9" stroke-dasharray="5 5"/>`;
 series.forEach(s=>{svg+=`<path d="${path(s.values)}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;const i=s.values.length-1;svg+=`<circle cx="${x(i)}" cy="${y(s.values[i])}" r="4" fill="${s.color}" stroke="#fff" stroke-width="2"/>`;});
 return svg+'</svg>';
}
async function replay() {
 const seq=++replaySequence, payload={patient_id:$('patient-select').value,family:$('model-select').value,hour:Number($('hour-slider').value),stress:$('stress-select').value,severity:$('severity-select').value};
 $('hour-title').textContent=payload.hour; $('hour-value').textContent=payload.hour;
 $('severity-select').disabled=payload.stress==='none'; $('replay-status').textContent='Calculating scores from the available history…';
 try {
  const result=await request('/api/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(seq!==replaySequence)return;
  const current=patients.find(p=>p.id===payload.patient_id),last=current.hours-1;
  $('timeline-chart').innerHTML=svgChart([{values:result.baseline_scores,color:'#087c74'},...(payload.stress==='none'?[]:[{values:result.scores,color:'#b77b1d'}])],last,result.threshold);
  $('current-score').textContent=percent(result.scores.at(-1));
  $('stress-legend').hidden=payload.stress==='none'; $('replay-ood').hidden=!result.out_of_distribution_illustration;
  $('replay-status').textContent=`Synthetic timeline · threshold ${result.threshold>1?'above 100% (no-alert policy)':percent(result.threshold)} · ${payload.stress==='none'?'original observations':payload.stress+', '+payload.severity+' severity'}. ${payload.stress==='delay'?'Measurement age is time since delayed availability, not specimen age.':''}`;
  $('measurements').innerHTML=result.measurements.map(m=>`<tr><td>${escapeHtml(m.label)}</td><td>${m.value===null?'—':escapeHtml(m.value)+' <span class="model-detail">'+escapeHtml(m.unit)+'</span>'}</td><td><span class="availability ${escapeHtml(m.status)}">${m.age_hours===null?'Unmeasured':m.age_hours===0?'Observed now':m.status==='stale'?'Stale · '+m.age_hours+' h':'Carried · '+m.age_hours+' h'}</span></td></tr>`).join('');
  const maximum=Math.max(...result.contributions.map(c=>Math.abs(c.value)),.001);
  $('contributions').innerHTML=result.contributions.length?result.contributions.map(c=>`<div class="contribution"><div class="contribution-title"><span class="contribution-name">${escapeHtml(c.feature)}</span><span class="contribution-value ${c.value>0?'positive':''}">${c.value>0?'+':''}${number(c.value,2)}</span></div><svg class="contribution-track" width="100%" height="5" role="presentation"><rect width="100%" height="5" fill="#f0f3f5"/><rect width="${(Math.abs(c.value)/maximum*100).toFixed(1)}%" height="5" rx="2" fill="${c.value>0?'#c99a54':'#40a293'}"/></svg></div>`).join(''):`<p class="empty">${escapeHtml(result.contribution_unavailable_reason || 'Feature attributions are not available for this model.')}</p>`;
 } catch(error) {if(seq===replaySequence)$('replay-status').textContent=error.message;}
}
function calibration(bins) {
 const w=420,h=250,left=47,bottom=33,top=12,right=20,x=v=>left+v*(w-left-right),y=v=>h-bottom-v*(h-bottom-top);
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Observed fraction against mean predicted score in hospital B">`;
 [0,.25,.5,.75,1].forEach(v=>{svg+=`<line x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}" stroke="#e6ecee"/><text x="${left-10}" y="${y(v)+4}" text-anchor="end">${v}</text><text x="${x(v)}" y="${h-14}" text-anchor="middle">${v}</text>`;});
 svg+=`<path d="M${x(0)},${y(0)} L${x(1)},${y(1)}" fill="none" stroke="#95a8af" stroke-dasharray="5 5"/>`;
 const points=(bins||[]).filter(b=>b.count>0&&finite(b.mean_probability)&&finite(b.observed_fraction));
 if(points.length)svg+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${x(p.mean_probability)},${y(p.observed_fraction)}`).join(' ')}" stroke="#087c74" stroke-width="2" fill="none"/>`;
 points.forEach(p=>svg+=`<circle cx="${x(p.mean_probability)}" cy="${y(p.observed_fraction)}" r="4" fill="#087c74" stroke="white" stroke-width="1.5"><title>${p.count} hours; predicted ${number(p.mean_probability)}, observed ${number(p.observed_fraction)}</title></circle>`);
 return svg+'</svg><div class="chart-caption">x: mean predicted score · y: observed label fraction</div>';
}
function reliability() {
 const family=$('report-model').value, model=report.models[family]; if(!model)return;
 const rows=[['Average precision', 'hourly_metrics','average_precision',number],['Brier score ↓','hourly_metrics','brier',number],['Challenge utility','challenge','challenge_utility',number],['Event sensitivity','pre_onset','event_sensitivity',percent],['Missed eligible events','pre_onset','missed_events',v=>finite(v)?v:'—'],['False alerts / patient-day','pre_onset','false_alerts_per_patient_day',v=>number(v,2)],['Median lead time (hours)','pre_onset','median_lead_time_hours',v=>number(v,1)],['Patients','root','patients',v=>finite(v)?v:'—']];
 $('transfer-table').innerHTML=rows.map(([label,section,key,format])=>`<tr><td>${label}</td>${['test','external'].map(split=>`<td>${format(section==='root'?model[split]?.[key]:model[split]?.[section]?.[key])}</td>`).join('')}</tr>`).join('');
 $('calibration-chart').innerHTML=calibration(model.external?.hourly_metrics?.reliability_bins);
 const stress=(report.stress_tests||[]).filter(s=>s.family===family);
 $('stress-table').innerHTML=stress.length?stress.map(s=>{const e=s.evaluation,condition={missing:'Missingness',missingness:'Missingness',delay:'Delay',noise:'Noise'}[s.kind]||s.kind;const severity=['missing','missingness'].includes(s.kind)?percent(s.severity):s.kind==='delay'?s.severity+' h':s.severity+' × train SD';return `<tr><td><strong>${escapeHtml(condition)}</strong><span class="model-detail">${escapeHtml(severity)}</span></td><td>${s.split==='external'?'B external':'A test'}</td><td>${number(e.hourly_metrics.average_precision)}</td><td>${number(e.hourly_metrics.brier)}</td><td>${percent(e.pre_onset.event_sensitivity)}</td><td>${number(e.pre_onset.false_alerts_per_patient_day,2)}</td></tr>`;}).join(''):'<tr><td colspan="6" class="empty">No stress experiment results for this model in the loaded run. Use Patient replay to inspect controlled synthetic perturbations.</td></tr>';
 const ci=model.external?.confidence_intervals||{};
 $('confidence').innerHTML=[['Average precision','average_precision'],['Event sensitivity','event_sensitivity'],['False alerts / day','false_alerts_per_patient_day']].map(([label,key])=>{const item=ci[key];return `<div class="interval-row"><span>${label}</span><strong>${finite(item?.lower)?number(item.lower)+'–'+number(item.upper):'Not estimable'}</strong></div>`;}).join('')+`<p class="footnote">${escapeHtml(report.metadata?.bootstrap_replicates??'Unspecified')} requested bootstrap replicates in this run. Undefined intervals are not replaced with zero.</p>`;
 const ablations=Object.entries(report.ablations||{}).filter(([,entry])=>entry.family===family);
 const ablationNames={no_trends:'Without trend features',no_missingness:'Without missingness features',values_only:'Recent values only',no_calibration:'Without calibration'};
 $('ablation-table').innerHTML=ablations.length?ablations.map(([name,entry])=>{const e=entry.external;return `<tr><td>${escapeHtml(ablationNames[name]||name)}</td><td>${number(e?.hourly_metrics?.average_precision)}</td><td>${number(e?.hourly_metrics?.brier)}</td><td>${percent(e?.pre_onset?.event_sensitivity)}</td><td>${number(e?.pre_onset?.false_alerts_per_patient_day,2)}</td></tr>`;}).join(''):'<tr><td colspan="5" class="empty">No component-ablation experiment is recorded for this model. Feature ablations use logistic regression; the calibration ablation uses the development-selected family.</td></tr>';
 const subgroupNames={age_lt65:'Age under 65',age_ge65:'Age 65 or older',age_missing:'Age not recorded',gender_0:'Recorded gender code 0',gender_1:'Recorded gender code 1',gender_missing_or_other:'Gender missing or other'};
 const groups=Object.entries(model.external?.subgroups||{});
 $('subgroup-table').innerHTML=groups.length?groups.map(([name,entry])=>`<tr><td>${escapeHtml(subgroupNames[name]||name)}</td><td>${entry.patients}</td><td>${entry.positive_patients}</td>${entry.suppressed?`<td colspan="3" class="na">Suppressed · ${escapeHtml(entry.reason||'insufficient patients or events')}</td>`:`<td>${number(entry.hourly_metrics?.average_precision)}</td><td>${number(entry.hourly_metrics?.brier)}</td><td>${percent(entry.pre_onset?.event_sensitivity)}</td>`}</tr>`).join(''):'<tr><td colspan="6" class="empty">Subgroup results are unavailable for this run.</td></tr>';
 const selection=report.selective_prediction?.[`${family}_external`];
 $('selective').innerHTML=selection?.results?.length?`<p>${selection.results.length} withholding settings were evaluated for this family in hospital B.</p><div class="table-wrap"><table><thead><tr><th>Coverage</th><th>Missed events</th><th>Events with withheld hours</th><th>Fully withheld events</th><th>Retained Brier</th><th>Withheld hours</th></tr></thead><tbody>${selection.results.map(r=>`<tr><td>${percent(r.coverage)}</td><td>${finite(r.missed_events)?r.missed_events:'—'}</td><td>${finite(r.eligible_events_with_any_withheld_warning_hour)?r.eligible_events_with_any_withheld_warning_hour:'—'}</td><td>${finite(r.eligible_events_with_all_warning_hours_withheld)?r.eligible_events_with_all_warning_hours_withheld:'—'}</td><td>${number(r.retained_brier)}</td><td>${finite(r.withheld_hours)?r.withheld_hours:'—'}</td></tr>`).join('')}</tbody></table></div><p class="footnote">Withheld-event counts use the six-hour pre-onset warning window: any withheld warning hour, or every warning hour withheld. Missed events treat withheld predictions as no alert. Retained Brier describes only predictions the model kept.</p>`:'<p>No selective-prediction experiment is recorded for this local demonstration. The reproducible benchmark command produces this analysis.</p>';
}
function setPatient() {
 const patient=patients.find(p=>p.id===$('patient-select').value);if(!patient)return;
 $('hour-slider').max=patient.hours-1;$('hour-slider').value=Math.min(Number($('hour-slider').value),patient.hours-1);$('last-hour').textContent=`Hour ${patient.hours-1}`;replay();
}
async function init() {
 page();
 try {
  [status,{report}, {patients}]=await Promise.all([request('/api/status'),request('/api/report'),request('/api/patients')]);
  const available=status.available_models;
  for(const id of ['model-select','report-model'])$(id).innerHTML=available.map(family=>`<option value="${escapeHtml(family)}">${names[family]||escapeHtml(family)}</option>`).join('');
  if(available.includes(report.selected_family))$('report-model').value=$('model-select').value=report.selected_family;
  $('patient-select').innerHTML=patients.map((p,i)=>`<option value="${escapeHtml(p.id)}">Patient ${String(i+1).padStart(3,'0')} · synthetic hospital ${escapeHtml(p.hospital)} · ${p.hours} h</option>`).join('');
  overview();reliability();setPatient();$('loading').hidden=true;
  if(status.runtime==='browser') { $('runtime-label').textContent='INTERACTIVE RESEARCH EDITION'; $('runtime-detail').textContent='Fitted models run in your browser. Replay uses generated timelines; aggregate benchmark results come from the recorded experiment.'; $('runtime-detail').hidden=false; $('replay-description').textContent='Only measurements at or before the selected hour enter the model. The challenge label includes pre-onset and post-onset hours. Browser perturbations use reproducible random draws distinct from the recorded benchmark stress experiments.'; }
 } catch(error) {$('loading').hidden=true;$('error').hidden=false;$('error').textContent=error.message+' Refresh after checking the local server configuration.';}
}
function reportCsv() {
 const headers=['source','complete_public_cohort','family','cohort','patients','patient_hours','average_precision','auroc','brier','event_sensitivity','eligible_events','missed_events','false_alerts_per_patient_day','median_lead_time_hours'];
 const rows=[];
 for(const [family,model] of Object.entries(report.models))for(const split of ['test','external']) { const e=model[split];if(!e)continue;rows.push([report.metadata?.source,report.metadata?.complete_public_cohort,family,split,e.patients,e.patient_hours,e.hourly_metrics?.average_precision,e.hourly_metrics?.auroc,e.hourly_metrics?.brier,e.pre_onset?.event_sensitivity,e.pre_onset?.eligible_events,e.pre_onset?.missed_events,e.pre_onset?.false_alerts_per_patient_day,e.pre_onset?.median_lead_time_hours]); }
 const cell=value=>`"${String(value??'').replaceAll('"','""')}"`;
 return [headers,...rows].map(row=>row.map(cell).join(',')).join('\r\n');
}
function downloadReport(format) {
 if(!report)return;
 const content=format==='json'?JSON.stringify(report,null,2):reportCsv();
 const blob=new Blob([content],{type:format==='json'?'application/json':'text/csv;charset=utf-8'});
 const href=URL.createObjectURL(blob),link=document.createElement('a');link.href=href;link.download=`clinical-ai-reliability-${report.metadata?.source||'report'}.${format}`;link.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
}
window.addEventListener('hashchange',page);
$('patient-select').addEventListener('change',setPatient);
for(const id of ['model-select','stress-select','severity-select'])$(id).addEventListener('change',replay);
let sliderTimer;$('hour-slider').addEventListener('input',()=>{clearTimeout(sliderTimer);$('hour-title').textContent=$('hour-value').textContent=$('hour-slider').value;sliderTimer=setTimeout(replay,120);});
$('report-model').addEventListener('change',reliability);
$('download-json').addEventListener('click',()=>downloadReport('json'));
$('download-csv').addEventListener('click',()=>downloadReport('csv'));
init();
