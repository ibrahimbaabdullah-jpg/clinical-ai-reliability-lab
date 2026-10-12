/* DOM-free rendering contract against an actual generated benchmark report.
 * Run: node tests/test_web_contract.cjs bundles/demo-v1/report.json
 * This checks JavaScript/data integration, not browser layout or accessibility.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const reportPath = process.argv[2] || 'bundles/demo-v1/report.json';
const actual = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
const script = fs.readFileSync(path.join(__dirname, '../clinical_lab/web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../clinical_lab/web/index.html'), 'utf8');
const knownIds = new Set(Array.from(html.matchAll(/\bid="([^"]+)"/g), match => match[1]));

async function render(report) {
  const elements = new Map();
  const listeners = new Map();
  let focused = null;
  function element(id) {
    assert.ok(knownIds.has(id), `JavaScript requests an absent element: ${id}`);
    if (!elements.has(id)) elements.set(id, {
      id, innerHTML: '', textContent: '', value: '', hidden: false,
      addEventListener(type, handler) { listeners.set(`${id}:${type}`, handler); },
      focus() { focused = id; }, setAttribute() {}, removeAttribute() {},
      classList: { toggle() {} }, hasChildNodes() { return !!this.innerHTML; }
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    document: {getElementById: element, querySelectorAll: selector => selector === '.page' ? ['overview','replay','reliability','methods'].map(element) : []},
    location: {hash: '#overview'}, window: {addEventListener() {}},
    setTimeout, clearTimeout, console,
    fetch: async url => {
      const routes = {
        '/api/status': {available_models: Object.keys(report.models), training_source: report.metadata.source},
        '/api/report': {report}, '/api/patients': {patients: []}
      };
      assert.ok(url in routes, `Unexpected fetch: ${url}`);
      return {ok: true, json: async () => routes[url]};
    }
  });
  vm.runInContext(script.replace(/init\(\);\s*$/, 'globalThis.initialized = init();'), context);
  await context.initialized;
  assert.equal(element('error').textContent, '', 'Rendering must complete without its error fallback');
  assert.equal(element('loading').hidden, true);
  const output = [...elements.values()].map(e => `${e.innerHTML} ${e.textContent}`).join('\n');
  assert.ok(!/\b(?:undefined|NaN)\b/.test(output), 'Rendered fields must not contain undefined or NaN');
  assert.equal((element('model-table').innerHTML.match(/>SELECTED</g) || []).length, 1);
  assert.equal(element('report-model').value, report.selected_family);
  assert.ok(element('calibration-chart').innerHTML.includes('<svg'), 'Calibration plot must render from report bins');
  return {element, context, listeners, focused: () => focused};
}

(async () => {
  const first = await render(actual);
  // Native Enter activation of an anchor emits click. Suppress its default
  // #main fragment navigation so the hash router preserves the current page.
  assert.match(html, /<a\b[^>]*id="skip-content"[^>]*href="#main"/);
  assert.match(html, /<main\b[^>]*id="main"[^>]*tabindex="-1"/);
  for (const route of ['overview','replay','reliability','methods']) {
    first.context.location.hash = `#${route}`;
    vm.runInContext('page()', first.context);
    const breadcrumb = first.element('breadcrumb').textContent;
    let prevented = false;
    const activate = first.listeners.get('skip-content:click');
    assert.equal(typeof activate, 'function', 'Skip link needs a navigation-preserving activation handler');
    activate({preventDefault() { prevented = true; }});
    if (!prevented) { first.context.location.hash = '#main'; vm.runInContext('page()', first.context); }
    assert.ok(prevented, 'Skip activation must prevent default fragment navigation');
    assert.equal(first.context.location.hash, `#${route}`);
    assert.equal(first.focused(), 'main');
    assert.equal(first.element(route).hidden, false);
    assert.equal(first.element('breadcrumb').textContent, breadcrumb);
  }
  const family = actual.selected_family;
  const stresses = actual.stress_tests.filter(s => s.family === family);
  for (const stress of stresses.filter(s => ['missing', 'missingness'].includes(s.kind))) {
    const expected = `Missingness</strong><span class="model-detail">${(stress.severity*100).toFixed(1)}%`;
    assert.ok(first.element('stress-table').innerHTML.includes(expected), 'Missingness severity must display percentage, never noise units');
  }
  const selection = actual.selective_prediction[`${family}_external`];
  assert.ok(selection.results.length > 0);
  for (const label of ['Events with withheld hours', 'Fully withheld events', 'Retained Brier']) {
    assert.ok(first.element('selective').innerHTML.includes(label), `Missing selective endpoint: ${label}`);
  }
  const changed = structuredClone(actual);
  changed.stress_tests.forEach(s => {if (s.kind === 'missing') s.kind = 'missingness';});
  const second = await render(changed);
  assert.equal(first.element('stress-table').innerHTML, second.element('stress-table').innerHTML, 'Both missingness aliases must render identically');
  const suppliedExpected = actual.metadata.source === 'synthetic' ? 'SYNTHETIC DEMONSTRATION' : actual.metadata.complete_public_cohort ? 'RETROSPECTIVE RESEARCH' : 'REDUCED-COHORT PRELIMINARY RUN';
  assert.equal(first.element('evidence-tag').textContent, suppliedExpected, 'The supplied report must disclose its own source and cohort status');
  changed.metadata.source = 'synthetic';
  changed.metadata.complete_public_cohort = false;
  const synthetic = await render(changed);
  assert.equal(synthetic.element('evidence-tag').textContent, 'SYNTHETIC DEMONSTRATION');
  changed.metadata.source = 'physionet-2019';
  changed.metadata.complete_public_cohort = false;
  const preliminary = await render(changed);
  assert.equal(preliminary.element('evidence-tag').textContent, 'REDUCED-COHORT PRELIMINARY RUN');
  changed.metadata.complete_public_cohort = true;
  const complete = await render(changed);
  assert.equal(complete.element('evidence-tag').textContent, 'RETROSPECTIVE RESEARCH');
  console.log(`Web rendering contract passed: ${reportPath}; skip-link route/focus preservation, missingness aliases, selection, calibration, selective endpoints, provenance.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
