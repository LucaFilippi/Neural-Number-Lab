const $ = s => document.querySelector(s); const $$ = s => [...document.querySelectorAll(s)];
const D = 28, N = 784, GOAL = 15;

/* ================= neural-network.js ================= */
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
const argmax = a => a.reduce((m, v, i) => v > a[m] ? i : m, 0);

function softmax(z) { // p_i = e^(z_i - max) / sum_j e^(z_j - max)
  let m = Math.max(...z), s = 0;
  for (let i = 0; i < z.length; i++) { z[i] = Math.exp(z[i] - m); s += z[i]; }
  for (let i = 0; i < z.length; i++) z[i] /= s;
}

class Net {
  constructor(sizes = [N, 64, 32, 9]) {
    this.sizes = sizes;
    this.init();
  }
  init() {
    this.W = []; this.b = [];
    for (let l = 1; l < this.sizes.length; l++) {
      const i = this.sizes[l - 1], o = this.sizes[l], s = Math.sqrt(2 / i), w = new Float32Array(i * o);
      for (let k = 0; k < w.length; k++) w[k] = gauss() * s;
      this.W.push(w);
      this.b.push(new Float32Array(o));
    }
  }
  params() {
    return this.sizes.slice(1).reduce((t, o, l) => t + o * this.sizes[l] + o, 0);
  }
  forward(x) {
    const A = [x], L = this.W.length;
    for (let l = 0; l < L; l++) {
      const i = this.sizes[l], o = this.sizes[l + 1], a = A[l], w = this.W[l], z = new Float32Array(o);
      for (let j = 0; j < o; j++) {
        let s = this.b[l][j]; const r = j * i;
        for (let k = 0; k < i; k++) s += w[r + k] * a[k];
        z[j] = s;
      }
      if (l < L - 1) { for (let j = 0; j < o; j++) if (z[j] < 0) z[j] = 0; }
      else softmax(z);
      A.push(z);
    }
    return A;
  }
  train(B, lr) {
    const L = this.W.length, gW = this.W.map(w => new Float32Array(w.length)), gb = this.b.map(b => new Float32Array(b.length));
    let loss = 0, ok = 0;
    for (const s of B) {
      const A = this.forward(s.x), p = A[L], t = s.l - 1;
      loss += -Math.log(p[t] + 1e-9);
      if (argmax(p) === t) ok++;
      let d = Float32Array.from(p); d[t] -= 1;
      for (let l = L - 1; l >= 0; l--) {
        const i = this.sizes[l], o = this.sizes[l + 1], a = A[l], w = this.W[l], nd = l > 0 ? new Float32Array(i) : null;
        for (let j = 0; j < o; j++) {
          const dj = d[j]; if (!dj) continue; gb[l][j] += dj; const r = j * i;
          for (let k = 0; k < i; k++) {
            gW[l][r + k] += dj * a[k];
            if (nd) nd[k] += w[r + k] * dj;
          }
        }
        if (nd) { for (let k = 0; k < i; k++) if (a[k] <= 0) nd[k] = 0; d = nd; }
      }
    }
    const sc = lr / B.length;
    for (let l = 0; l < L; l++) {
      for (let k = 0; k < gW[l].length; k++) this.W[l][k] -= sc * gW[l][k];
      for (let k = 0; k < gb[l].length; k++) this.b[l][k] -= sc * gb[l][k];
    }
    return { loss: loss / B.length, ok };
  }
}

/* ================= storage.js ================= */
const b64 = u8 => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const f32 = a => b64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const unf32 = s => new Float32Array(unb64(s).buffer);

function pack() {
  return {
    samples: S.map(s => ({ l: s.l, x: b64(Uint8Array.from(s.x, v => Math.round(v * 255))) })),
    net: { sizes: net.sizes, W: net.W.map(f32), b: net.b.map(f32) },
    hist,
    cfg
  };
}

function unpack(o) {
  S = o.samples.map(s => ({ l: s.l, x: Float32Array.from(unb64(s.x), v => v / 255) }));
  net = new Net(o.net.sizes);
  net.W = o.net.W.map(unf32);
  net.b = o.net.b.map(unf32);
  hist = o.hist || [];
  Object.assign(cfg, o.cfg || {});
}

function save() { try { localStorage.setItem('nnl', JSON.stringify(pack())); } catch (e) { } }
function load() { try { const t = localStorage.getItem('nnl'); if (t) unpack(JSON.parse(t)); } catch (e) { } }

/* ================= drawing.js ================= */
function pad(cv, sw, onEnd) {
  const g = cv.getContext('2d'); let down = false, px, py, ink = false;
  const clr = () => { g.fillStyle = '#fff'; g.fillRect(0, 0, 280, 280); ink = false; }; clr();
  const pos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * 280 / r.width, (e.clientY - r.top) * 280 / r.height]; };
  const seg = (x, y) => { g.strokeStyle = '#111'; g.lineCap = g.lineJoin = 'round'; g.lineWidth = +sw.value; g.beginPath(); g.moveTo(px, py); g.lineTo(x + .01, y); g.stroke(); px = x; py = y; ink = true; };
  cv.onpointerdown = e => { e.preventDefault(); down = true; cv.setPointerCapture(e.pointerId);[px, py] = pos(e); seg(px, py); };
  cv.onpointermove = e => { if (down) { e.preventDefault(); seg(...pos(e)); } };
  const up = () => { if (down) { down = false; onEnd && onEnd(); } };
  cv.onpointerup = up; cv.onpointercancel = up;
  return { clr, has: () => ink, cv };
}

function prep(cv) {
  const d = cv.getContext('2d').getImageData(0, 0, 280, 280).data, ink = new Float32Array(78400);
  let x0 = 280, y0 = 280, x1 = -1, y1 = -1;
  for (let i = 0; i < 78400; i++) {
    const v = 1 - d[i * 4] / 255; ink[i] = v;
    if (v > .15) {
      const x = i % 280, y = (i / 280) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h), ox = 14 - w * s / 2, oy = 14 - h * s / 2, sum = new Float32Array(N), cnt = new Float32Array(N);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const u = Math.min(27, Math.max(0, Math.floor(ox + (x - x0) * s))), v = Math.min(27, Math.max(0, Math.floor(oy + (y - y0) * s)));
    sum[v * D + u] += ink[y * 280 + x]; cnt[v * D + u]++;
  }
  const o = new Float32Array(N); let m = 0;
  for (let i = 0; i < N; i++) { o[i] = cnt[i] ? sum[i] / cnt[i] : 0; if (o[i] > m) m = o[i]; }
  let M = 0, cx = 0, cy = 0;
  for (let i = 0; i < N; i++) { o[i] /= m || 1; M += o[i]; cx += o[i] * (i % D); cy += o[i] * ((i / D) | 0); }
  const dx = Math.round(13.5 - cx / M), dy = Math.round(13.5 - cy / M), r = new Float32Array(N);
  for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) {
    const nx = x + dx, ny = y + dy;
    if (nx >= 0 && nx < D && ny >= 0 && ny < D) r[ny * D + nx] = o[y * D + x];
  }
  return r;
}

function showPrev(x) {
  const g = $('#prev').getContext('2d'), im = g.createImageData(D, D);
  for (let i = 0; i < N; i++) { const v = 255 - x[i] * 255; im.data.set([v, v, v, 255], i * 4); }
  g.putImageData(im, 0, 0);
}

/* ================= training.js + state ================= */
let S = [], net = new Net(), hist = [], cfg = { lr: 0.05, ep: 30, bs: 8, auto: true }, cur = 1, busy = false, lastA = null, lastX = null;
const pause = () => new Promise(r => setTimeout(r, 0));

async function trainRun(E) {
  if (busy || !S.length) return; busy = true; const n = S.length;
  for (let e = 0; e < E; e++) {
    const idx = [...S.keys()].sort(() => Math.random() - .5); let loss = 0, ok = 0;
    for (let i = 0; i < n; i += cfg.bs) {
      const r = net.train(idx.slice(i, i + cfg.bs).map(k => S[k]), cfg.lr), m = Math.min(cfg.bs, n - i);
      loss += r.loss * m; ok += r.ok;
    }
    hist.push({ l: loss / n, a: ok / n }); if (hist.length > 1500) hist.shift();
    $('#st1').textContent = `Training… epoch ${e + 1} of ${E}`; ui(); await pause();
  }
  busy = false; $('#st1').textContent = `Done. Trained on ${n} examples.`; save(); ui();
}

function addSample(l, x) { S.push({ l, x }); save(); ui(); if (cfg.auto) trainRun(20); }

/* ================= visuals ================= */
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function chart(cv, vals, color, fixedMax, title) {
  const g = cv.getContext('2d'), W = cv.width, H = cv.height; g.clearRect(0, 0, W, H);
  g.strokeStyle = css('--ln'); g.fillStyle = css('--mu'); g.font = '12px sans-serif';
  for (let i = 0; i <= 2; i++) { const y = 14 + (H - 28) * i / 2; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  g.fillText(title, 8, 12);
  if (vals.length < 2) { g.fillText('no data yet', W / 2 - 30, H / 2 + 4); return; }
  const mx = fixedMax || Math.max(...vals) * 1.1 || 1; g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
  vals.forEach((v, i) => { const x = i / (vals.length - 1) * (W - 4) + 2, y = H - 14 - (v / mx) * (H - 28); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
}

function drawNet(cv, A) {
  const g = cv.getContext('2d'), W = cv.width, H = cv.height, sz = net.sizes, show = [0, 10, 8, 9], X = [.1, .37, .63, .88].map(v => v * W);
  g.clearRect(0, 0, W, H);
  const ys = (c, k) => H * .12 + H * .68 * k / (show[c] - 1), idx = (c, k) => Math.round(k * (sz[c] - 1) / (show[c] - 1));
  const mxs = sz.map((_, c) => A && c > 0 && c < 3 ? Math.max(...A[c]) || 1 : 1);
  for (let c = 1; c < 4; c++) for (let j = 0; j < show[c]; j++) for (let k = 0; k < (c == 1 ? 1 : show[c - 1]); k++) {
    let a = .12, col = css('--mu');
    if (c > 1) { const w = net.W[c - 1][idx(c, j) * sz[c - 1] + idx(c - 1, k)]; a = Math.min(.8, Math.abs(w) * 1.6); col = w > 0 ? css('--ac') : css('--ng'); }
    g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); g.moveTo(c == 1 ? X[0] + 36 : X[c - 1], c == 1 ? H * .46 : ys(c - 1, k)); g.lineTo(X[c], ys(c, j)); g.stroke();
  }
  g.globalAlpha = 1;
  const bs = 72, gi = g.createImageData(D, D), im = document.createElement('canvas'); im.width = im.height = D;
  for (let i = 0; i < N; i++) { const v = A ? 255 - A[0][i] * 255 : 235; gi.data.set([v, v, v, 255], i * 4); }
  im.getContext('2d').putImageData(gi, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(im, X[0] - 36, H * .46 - 36, bs, bs);
  const P = A ? A[3] : null, win = P ? argmax(P) : -1;
  for (let c = 1; c < 4; c++) for (let j = 0; j < show[c]; j++) {
    const x = X[c], y = ys(c, j), out = c == 3, r = out ? 11 : 7;
    const act = !A ? 0 : out ? P[j] : Math.min(1, A[c][idx(c, j)] / mxs[c]);
    g.beginPath(); g.arc(x, y, r, 0, 7); g.fillStyle = css('--bg'); g.fill();
    g.globalAlpha = .15 + .85 * act; g.fillStyle = out && j == win ? css('--wm') : css('--ac'); g.fill(); g.globalAlpha = 1;
    g.lineWidth = out && j == win ? 3 : 1; g.strokeStyle = out && j == win ? css('--wm') : css('--ln'); g.stroke();
    if (out) { g.fillStyle = j == win ? css('--wm') : css('--mu'); g.font = (j == win ? '800 16px' : '14px') + ' sans-serif'; g.fillText(j + 1, x + 18, y + 5); }
  }
  g.fillStyle = css('--mu'); g.font = '12px sans-serif'; g.textAlign = 'center';
  ['28×28 input', sz[1] + ' neurons', sz[2] + ' neurons', '9 outputs'].forEach((t, c) => g.fillText(t, X[c], H - 8)); g.textAlign = 'left';
}

function ui() {
  const cnt = Array(10).fill(0); S.forEach(s => cnt[s.l]++);
  $$('#digits button').forEach(b => { const n = +b.dataset.d; b.querySelector('i').textContent = cnt[n]; b.classList.toggle('on', n === cur); b.style.setProperty('--f', Math.min(1, cnt[n] / GOAL)); });
  $('#cur').textContent = cur;   const h = hist.at(-1);   $$('.s-n').forEach(e => e.textContent = S.length); $$('.s-e').forEach(e => e.textContent = hist.length);$$('.s-l').forEach(e => e.textContent = h ? h.l.toFixed(2) : '–'); $$('.s-a').forEach(e => e.textContent = h ? (h.a * 100).toFixed(1) + '\%' : '–');   $$('.charts').forEach(c => {
    const cs = c.querySelectorAll('canvas');
    chart(cs[0], hist.map(x => x.l), css('--ng'), 0, 'Loss (lower is better)'); chart(cs[1], hist.map(x => x.a), css('--ac'), 1, 'Accuracy (higher is better)');
  });
  const low = [1, 2, 3, 4, 5, 6, 7, 8, 9].sort((a, b) => cnt[a] - cnt[b])[0];
  $('#hint').textContent = cnt[low] >= GOAL ? 'Every number has plenty of examples. Try unusual styles: slanted, thin, wide, messy.' : `Vary your handwriting. Fewest examples so far: ${low} (${cnt[low]}). Aim for ${GOAL}+ of each.`;
  $('#untrained').textContent = hist.length ? '' : 'The AI has not been trained yet, so its guesses are random.';
  $('#arch').textContent = `${net.sizes.join(' → ')} · ReLU hidden layers · softmax output · ${net.params().toLocaleString()} weights and biases · learning rate ${cfg.lr}, batch ${cfg.bs}`;
  drawNet($('#net2'), lastA);
}

/* ================= app.js ================= */
$('#digits').innerHTML = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(d => `<button data-d="${d}">${d}<i>0</i></button>`).join('');
$('#teach').innerHTML = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(d => `<button class="b" data-t="${d}" style="padding:4px 10px">${d}</button>`).join('');
$('#bars').innerHTML = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(d => `<div class="bar"><span>${d}</span><div><u></u></div><span>–</span></div>`).join('');
$$('.charts').forEach(c => c.innerHTML = '<canvas class="ch" width="300" height="150"></canvas><canvas class="ch" width="300" height="150"></canvas>');
load(); $('#lr').value = cfg.lr; $('#ep').value = cfg.ep; $('#bs').value = cfg.bs; $('#auto').checked = cfg.auto;

$('#tabs').onclick = e => {
  const p = e.target.dataset.p; if (!p) return;
  $$('#tabs button').forEach(b => b.classList.toggle('on', b === e.target)); $$
('.page').forEach(s => s.classList.toggle('on', s.id === p)); ui();
};
$('#digits').onclick = e => { const b = e.target.closest('button'); if (b) { cur = +b.dataset.d; ui(); } };
const p1 = pad($('#c1'), $('#sw1'), () => { const x = prep($('#c1')); if (x) showPrev(x); }), p2 = pad($('#c2'), $('#sw2'));
$('#clr1').onclick = p1.clr; $('#clr2').onclick = p2.clr;
$('#submit').onclick = () => {
  const x = prep($('#c1')); if (!x) { $('#st1').textContent = 'Draw a number first, then submit it.'; return; }
  addSample(cur, x); p1.clr();
};
$('#trainBtn').onclick = $('#trainBtn2').onclick = () => { if (!S.length) { $('#st1').textContent = 'Add some training examples first.'; return; } trainRun(cfg.ep); };
$('#auto').onchange = e => { cfg.auto = e.target.checked; save(); };
[['lr', 'lr', parseFloat], ['ep', 'ep', parseInt], ['bs', 'bs', parseInt]].forEach(([id, k, f]) => $('#' + id).onchange = e => { const v = f(e.target.value); if (v > 0) cfg[k] = v; e.target.value = cfg[k]; save(); ui(); });
$('#recog').onclick = () => {
  const x = prep($('#c2')); if (!x) return; lastX = x; lastA = net.forward(x); const P = lastA[3], w = argmax(P);
  $('#pred').textContent = w + 1; $('#conf').textContent = (P[w] * 100).toFixed(1) + '\%';   $$('#bars .bar').forEach((r, i) => { r.classList.toggle('top', i === w); r.querySelector('u').style.width = P[i] * 100 + '%'; r.lastElementChild.textContent = (P[i] * 100).toFixed(1) + '%'; });
  drawNet($('#net1'), lastA); drawNet($('#net2'), lastA);
};
$('#teach').onclick = e => { const t = e.target.dataset.t; if (t && lastX) { addSample(+t, lastX); $('#st1').textContent = `Added your drawing as a ${t}.`; } };
$('#resetAI').onclick = () => { if (busy) return; net = new Net(); hist = []; lastA = null; save(); ui(); $('#st1').textContent = 'AI reset. It knows nothing again.'; };
$('#resetAll').onclick = () => { if (busy || !confirm('Delete all examples and the trained AI?')) return; S = []; net = new Net(); hist = []; lastA = null; try { localStorage.removeItem('nnl'); } catch (e) { } ui(); };

/* === Exportação via Download de arquivo JSON === */
$('#exp').onclick = () => {
  const t = JSON.stringify(pack(), null, 2);
  $('#io').value = t;
  $('#io').select();

  // Criação do Blob e acionamento do download automático
  const blob = new Blob([t], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'neural_model.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

$('#imp').onclick = () => {
  try {
    unpack(JSON.parse($('#io').value)); lastA = null; save(); ui(); $('#io').value = '';
  } catch (e) {
    $('#io').value = ''; $('#io').placeholder = 'Could not read that data. Paste a full export.';
  }
};
$('#file').onchange = e => {
  const f = e.target.files[0];
  if (f) {
    const r = new FileReader();
    r.onload = () => { $('#io').value = r.result; };
    r.readAsText(f);
  }
};
ui();