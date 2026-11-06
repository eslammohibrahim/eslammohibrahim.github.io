// Hero animation: two clouds of points and a logistic regression learning to separate them,
// one gradient step per frame. When it has converged it waits a bit and starts over with new data.

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const canvas = document.getElementById("boundary");
const ctx = canvas.getContext("2d");
const hudStep = document.getElementById("hud-step");
const hudAcc = document.getElementById("hud-acc");

let W = 0, H = 0, dpr = 1;
let points = [];
let w = [0, 0, 0];  // weights for x, y and the bias
let step = 0;
let pauseUntil = 0;

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.clientWidth;
  H = canvas.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function gaussian() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function newData() {
  // coordinates live in [-1, 1]; the canvas maps them to the right side of the hero
  const angle = Math.random() * Math.PI;
  const cx = Math.cos(angle) * 0.42, cy = Math.sin(angle) * 0.42;
  points = [];
  for (let i = 0; i < 140; i++) {
    const label = i % 2;
    const s = label ? 1 : -1;
    points.push({
      x: s * cx + gaussian() * 0.3,
      y: s * cy + gaussian() * 0.3,
      label,
      born: performance.now() + Math.random() * 600,
    });
  }
  // start from a random wrong-ish line so the learning is visible
  w = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 0.6 - 0.3];
  step = 0;
}

const sigmoid = z => 1 / (1 + Math.exp(-z));

function trainStep(lr = 0.35) {
  const g = [0, 0, 0];
  for (const p of points) {
    const err = sigmoid(w[0] * p.x + w[1] * p.y + w[2]) - p.label;
    g[0] += err * p.x;
    g[1] += err * p.y;
    g[2] += err;
  }
  for (let k = 0; k < 3; k++) w[k] -= lr * g[k] / points.length;
  step++;
}

function accuracy() {
  let ok = 0;
  for (const p of points) ok += ((w[0] * p.x + w[1] * p.y + w[2] > 0) ? 1 : 0) === p.label;
  return ok / points.length;
}

// map data space to screen space: the plot sits on the right on wide screens, behind the text on small ones
function view() {
  const wide = W > 1000;
  const size = wide ? Math.min(H * 0.7, W * 0.36) : Math.min(W, H) * 0.9;
  const r = size / 2;
  const cx = wide ? W - Math.max(56, W * 0.05) - r : W * 0.5;
  const cy = H * 0.52;
  return { cx, cy, r };
}

function draw(now) {
  const { cx, cy, r } = view();
  const accent = css("--accent"), accent2 = css("--accent-2"), line = css("--line"), text = css("--text");
  ctx.clearRect(0, 0, W, H);

  // faint grid
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.5;
  for (let i = -4; i <= 4; i++) {
    ctx.beginPath(); ctx.moveTo(cx + (i / 4) * r, cy - r); ctx.lineTo(cx + (i / 4) * r, cy + r); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - r, cy + (i / 4) * r); ctx.lineTo(cx + r, cy + (i / 4) * r); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // shaded side of the decision boundary: sample a coarse grid of the probability
  const cells = 28;
  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      const x = -1 + (2 * (i + 0.5)) / cells, y = -1 + (2 * (j + 0.5)) / cells;
      const p = sigmoid(w[0] * x + w[1] * y + w[2]);
      ctx.fillStyle = p > 0.5 ? accent : accent2;
      ctx.globalAlpha = Math.abs(p - 0.5) * 0.16;
      ctx.fillRect(cx + x * r - r / cells, cy - y * r - r / cells, (2 * r) / cells + 0.5, (2 * r) / cells + 0.5);
    }
  }
  ctx.globalAlpha = 1;

  // the boundary line itself: w0*x + w1*y + w2 = 0
  if (Math.abs(w[1]) > 1e-6 || Math.abs(w[0]) > 1e-6) {
    let x1, y1, x2, y2;
    if (Math.abs(w[1]) > Math.abs(w[0])) {
      x1 = -1.2; x2 = 1.2;
      y1 = -(w[0] * x1 + w[2]) / w[1]; y2 = -(w[0] * x2 + w[2]) / w[1];
    } else {
      y1 = -1.2; y2 = 1.2;
      x1 = -(w[1] * y1 + w[2]) / w[0]; x2 = -(w[1] * y2 + w[2]) / w[0];
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(cx - r, cy - r, 2 * r, 2 * r);
    ctx.clip();
    ctx.strokeStyle = text;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(cx + x1 * r, cy - y1 * r);
    ctx.lineTo(cx + x2 * r, cy - y2 * r);
    ctx.stroke();
    ctx.restore();
  }

  // points pop in, and wrongly classified ones get a ring
  ctx.save();
  ctx.beginPath();
  ctx.rect(cx - r, cy - r, 2 * r, 2 * r);
  ctx.clip();
  for (const p of points) {
    const t = Math.min(Math.max((now - p.born) / 400, 0), 1);
    if (t === 0) continue;
    const px = cx + p.x * r, py = cy - p.y * r;
    const wrong = ((w[0] * p.x + w[1] * p.y + w[2] > 0) ? 1 : 0) !== p.label;
    ctx.fillStyle = p.label ? accent : accent2;
    ctx.beginPath();
    ctx.arc(px, py, 3.2 * (0.6 + 0.4 * t), 0, Math.PI * 2);
    ctx.globalAlpha = t;
    ctx.fill();
    if (wrong) {
      ctx.strokeStyle = text;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

function loop(now) {
  if (now > pauseUntil) {
    if (step < 260) {
      trainStep();
      if (step === 260) pauseUntil = now + 2600;
    } else {
      newData();
    }
  }
  hudStep.textContent = step;
  hudAcc.textContent = Math.round(accuracy() * 100) + "%";
  draw(now);
  requestAnimationFrame(loop);
}

resize();
window.addEventListener("resize", resize);
newData();
if (reduceMotion) {
  for (let i = 0; i < 260; i++) trainStep();
  points.forEach(p => (p.born = 0));
  hudStep.textContent = step;
  hudAcc.textContent = Math.round(accuracy() * 100) + "%";
  draw(performance.now());
} else {
  pauseUntil = performance.now() + 900;  // let the points appear before training starts
  requestAnimationFrame(loop);
}

// nav border once the page scrolls
const nav = document.querySelector(".nav");
window.addEventListener("scroll", () => nav.classList.toggle("scrolled", window.scrollY > 10), { passive: true });

// cards fade in when they come into view, and their numbers count up
function countUp(el) {
  const target = parseFloat(el.dataset.count);
  const decimals = parseInt(el.dataset.decimals || "0", 10);
  const prefix = el.dataset.prefix || "", suffix = el.dataset.suffix || "";
  if (reduceMotion) { el.textContent = prefix + target.toFixed(decimals) + suffix; return; }
  const start = performance.now(), dur = 1100;
  const tick = now => {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const observer = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    e.target.querySelectorAll("[data-count]").forEach(countUp);
    observer.unobserve(e.target);
  }
}, { threshold: 0.25 });
document.querySelectorAll("[data-reveal]").forEach(el => observer.observe(el));

// soft light that follows the mouse over a card
document.querySelectorAll(".card").forEach(card => {
  card.addEventListener("pointermove", e => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});
