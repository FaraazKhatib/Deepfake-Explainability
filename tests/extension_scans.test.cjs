const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const directory = path.join(__dirname, '..', 'browser-extension');
const worker = readFileSync(path.join(directory, 'background.js'), 'utf8');
const popup = readFileSync(path.join(directory, 'popup', 'popup.js'), 'utf8');

function harness(initialState) {
  let listener;
  const records = { vt_state: initialState };
  const requests = [];
  const notifications = [];
  const badges = [];
  const chrome = {
    runtime: { getURL: x => x, onMessage: { addListener: fn => { listener = fn; } } },
    storage: { local: {
      get: async key => ({ [key]: records[key] }),
      set: async patch => Object.assign(records, patch),
      remove: async key => { delete records[key]; },
    } },
    action: {
      setBadgeText: async ({ text }) => badges.push(text),
      setBadgeBackgroundColor: async () => {},
    },
    notifications: {
      create: async id => notifications.push(id),
      onClicked: { addListener() {} },
    },
  };
  // Deliberately let an aborted fetch settle later, to exercise stale callbacks.
  const fetch = (url, options) => new Promise((resolve, reject) => {
    requests.push({ url, options, reject, finish: summary => resolve({
      ok: true, json: async () => ({ verdict: 'Supported', summary }),
    }) });
  });
  vm.runInNewContext(worker, { chrome, fetch, AbortController, URL, setTimeout, clearTimeout, console });
  const send = msg => new Promise(resolve => listener(msg, {}, resolve));
  const start = (scanId, args = {}) => send({ action: 'START_ANALYSIS', scanId, payload: {
    text: 'An example claim was submitted.', singleClaim: true, ...args,
  } });
  const state = async () => (await send({ action: 'GET_SCAN_STATE' })).state;
  return { chrome, records, requests, notifications, badges, send, start, state };
}

test('replacement keeps its result and notification after an old response or stale cancellation', async () => {
  const h = harness();
  const first = h.start('first');
  await h.state();
  const second = h.start('second', { strictMode: true, singleClaim: false, text: 'a'.repeat(5200), sourceLanguage: 'mr' });
  assert.equal((await h.state()).scanId, 'second');
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[0].options.signal.aborted, true);
  const payload = JSON.parse(h.requests[1].options.body);
  assert.equal(payload.max_claims, 3);
  assert.equal(payload.text.length, 5000);
  assert.equal(payload.source_language, 'mr');
  assert.equal((await h.send({ action: 'CANCEL_ANALYSIS', scanId: 'first' })).status, 'superseded');
  h.requests[1].finish('new');
  await second;
  h.requests[0].finish('old');
  await first;
  assert.equal((await h.state()).resultData.summary, 'new');
  assert.deepEqual(h.notifications, ['vt_scan_done_second']);
  assert.equal(h.badges.at(-1), 'TRUE');
});

test('reset ignores a late abort error and leaves storage and badge cleared', async () => {
  const h = harness();
  const scan = h.start('cancelled');
  await h.state();
  await h.send({ action: 'CANCEL_ANALYSIS', scanId: 'cancelled' });
  assert.ok(h.requests[0].options.signal.aborted);
  h.requests[0].reject(new DOMException('cancelled', 'AbortError'));
  await scan;
  assert.equal(await h.state(), null);
  assert.deepEqual(h.notifications, []);
  assert.equal(h.badges.at(-1), '');
});

test('a reopened popup recovers an interrupted legacy scan and can run a new check', async () => {
  const h = harness({ vt_status: 'scanning', extractedText: 'legacy' });
  assert.equal((await h.state()).vt_status, 'error');
  const scan = h.start('new');
  await h.state();
  h.requests[0].reject(new TypeError('Failed to fetch'));
  await scan;
  assert.match((await h.state()).error, /Cannot connect/);
});

test('popup sends one check and reopening reads the completed worker result', async () => {
  const h = harness();
  const changes = new Set();
  const originalSet = h.chrome.storage.local.set;
  h.chrome.storage.local.set = async patch => {
    await originalSet(patch);
    if (patch.vt_state) for (const fn of changes) fn({ vt_state: { newValue: patch.vt_state } }, 'local');
  };
  h.chrome.storage.onChanged = { addListener: fn => changes.add(fn) };
  h.chrome.runtime.sendMessage = h.send;
  function openPopup(feedbackResponse) {
    const elements = new Map();
    function get(selector) {
      if (!elements.has(selector)) elements.set(selector, {
        style: {}, classList: { add() {}, remove() {}, toggle() {} },
        querySelector: () => null, appendChild() {}, value: '',
        addEventListener(event, fn) { this[event] = fn; },
      });
      return elements.get(selector);
    }
    const document = { createElement: () => ({}), readyState: 'loading', querySelector: get, querySelectorAll: () => [],
      addEventListener(type, fn) { this.init = fn; } };
    const localStorage = { getItem() {}, setItem() {} };
    const directRequests = [];
    vm.runInNewContext(popup, { document, chrome: h.chrome, localStorage, crypto, AbortSignal, URL, window: { innerWidth: 450 },
      setTimeout: (...args) => setTimeout(...args).unref(), clearTimeout, console, fetch: (...args) => { directRequests.push(args); if (feedbackResponse) return feedbackResponse(...args); throw Error('Unexpected direct fetch'); } });
    return { get, init: () => document.init(), directRequests };
  }
  const first = openPopup();
  await first.init();
  first.get('#inputText').value = 'A library has opened today.';
  const submitted = first.get('#analyzeBtn').click({ preventDefault() {} });
  await h.state();
  assert.equal(h.requests.length, 1);
  assert.equal(JSON.parse(h.requests[0].options.body).max_claims, 1);
  assert.deepEqual(first.directRequests, []);
  // Deliver the result to storage even when no popup listener exists.
  changes.clear();
  h.requests[0].finish('A supported claim');
  await submitted;
  const saved = (await h.state());
  saved.resultData.claims = [{ claim: 'A library has opened today.' }];
  let online = false;
  const reopened = openPopup(async (url, options) => {
    assert.ok(url.endsWith('/api/v1/text/feedback'));
    const report = JSON.parse(options.body);
    assert.equal(report.scan_id, saved.scanId);
    assert.equal(report.original_verdict, 'Supported');
    assert.equal(report.suggested_verdict, 'contradicted');
    assert.deepEqual(report.claims, ['A library has opened today.']);
    return { ok: online, json: async () => ({ status: 'saved', scan_id: saved.scanId }) };
  });
  await reopened.init();
  assert.equal(reopened.get('#summaryText').textContent, 'A supported claim');
  assert.equal(reopened.get('#statusBadge').textContent, 'Done');
  assert.equal(h.requests.length, 1);
  assert.equal(reopened.get('#feedbackPanel').hidden, false);
  reopened.get('#feedbackVerdict').value = 'contradicted';
  const form = reopened.get('#feedbackForm');
  await form.submit({ preventDefault() {} });
  assert.match(reopened.get('#feedbackStatus').textContent, /retry/);
  online = true;
  await form.submit({ preventDefault() {} });
  assert.match(reopened.get('#feedbackStatus').textContent, /saved for review/);
});
