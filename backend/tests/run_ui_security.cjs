// Uses the same browser fixture in jsdom, without external resources or Gemini.
// Install jsdom@26.1.0 in an isolated directory and expose it via NODE_PATH.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');

const ui = new JSDOM(fs.readFileSync(path.join(__dirname, '../ui/index.html'), 'utf8'), {
  url: 'http://127.0.0.1:8765/ui/index.html', runScripts: 'outside-only',
});
const harness = new JSDOM(fs.readFileSync(path.join(__dirname, 'ui_security.html'), 'utf8'), {
  url: 'http://127.0.0.1:8765/tests/ui_security.html', runScripts: 'outside-only',
});

(async () => {
  try {
    for (const script of ui.window.document.querySelectorAll('script:not([src])')) ui.window.eval(script.textContent);
    harness.window.loadUiForTest = async () => ui.window;
    for (const script of harness.window.document.querySelectorAll('script')) harness.window.eval(script.textContent);
    for (let i = 0; i < 100 && !harness.window.document.documentElement.dataset.testStatus; i++) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    console.log(harness.window.document.getElementById('summary').textContent);
    for (const item of harness.window.document.querySelectorAll('#results li')) console.log(item.textContent);
    assert.equal(harness.window.document.documentElement.dataset.testStatus, 'passed');
    assert.equal(harness.window.document.querySelectorAll('#results li').length, 9);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  } finally {
    ui.window.close();
    harness.window.close();
  }
})();
