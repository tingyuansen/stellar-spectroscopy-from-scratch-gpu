/* Render an executed .ipynb into a content/<slug>.html fragment using the EXACT
   assets/render.js pipeline (so the output is byte-faithful to the Agents book).
   Usage: node render_fragment.js <nb.ipynb> <out.html> <n> <title> <lecturer> <affil> */
const fs = require('fs');
const path = require('path');

// Provide the browser globals render.js expects, in Node.
global.window = global;
global.marked = require('marked');                       // window.marked.parse
global.katex = require('katex');                         // window.katex.renderToString
const _hljs = require('highlight.js');
global.hljs = _hljs.default || _hljs;                    // window.hljs.highlight / highlightAuto / getLanguage

require(path.resolve(__dirname, '..', 'assets', 'render.js'));  // defines window.BookRender

const [, , nbPath, outPath, n, title, lecturer, affil] = process.argv;
const raw = fs.readFileSync(nbPath, 'utf8');
const meta = { n: parseInt(n, 10), type: 'ipynb', title, lecturer, affil,
               slug: path.basename(outPath).replace(/\.html$/, '') };
const { objHtml, body } = window.BookRender.renderChapter(meta, raw);   // objHtml = callout, body = cells
fs.writeFileSync(outPath, objHtml + body);
console.log(`rendered ${path.basename(outPath)}  (${(objHtml + body).length} bytes, callout=${objHtml.length > 0})`);
