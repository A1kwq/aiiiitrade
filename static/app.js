// Crypto AI Trader - Dashboard UI logic
let equityChart = null;

// ── Refresh all data ─────────────────────────────
async function refreshAll() {
  // Stats
  try {
    const res = await fetch('/api/stats');
    const d = await res.json();
    if (d.balances) {
      document.getElementById('stat-balance').textContent = d.balances.total.toLocaleString('ru-RU', {maximumFractionDigits:0});
      document.getElementById('stat-pnl').textContent = d.pnl_24h.toLocaleString('ru-RU', {maximumFractionDigits:2});
    }
  } catch(e) {}

  // Positions
  try {
    const res = await fetch('/api/positions');
    const d = await res.json();
    renderPositions(d.positions || []);
  } catch(e) {}

  // Signals
  try {
    const res = await fetch('/api/signals');
    const d = await res.json();
    renderSignals(d.signals || []);
  } catch(e) {}

  // Equity curve
  try {
    const res = await fetch('/api/equity');
    const d = await res.json();
    renderEquity(d.equity || []);
  } catch(e) {}
}

// ── Positions table ───────────────────────────────
function renderPositions(positions) {
  const tbody = document.getElementById('positions-body');
  if (!tbody) return;
  if (!positions.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:20px">Нет открытых позиций</td></tr>';
    return;
  }
  let html = '';
  positions.forEach(p => {
    const pnlClass = p.pnl >= 0 ? 'green' : 'red';
    html += `<tr>
      <td><strong>${p.symbol}</strong></td>
      <td class="${p.side === 'buy' ? 'buy' : 'sell'}">${p.side === 'buy' ? 'Long' : 'Short'}</td>
      <td>${Number(p.price).toFixed(2)}</td>
      <td>${Number(p.size).toFixed(2)}</td>
      <td class="${pnlClass}">${p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(2)} ₽</td>
      <td><span class="badge badge-${p.status === 'open' ? 'green' : 'red'}">${p.status}</span></td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

// ── Signals list ───────────────────────────────────
function renderSignals(signals) {
  const el = document.getElementById('signals-list');
  if (!el) return;
  if (!signals.length) {
    el.innerHTML = '<p class="muted">Сигналов пока нет. Запустите бэктест.</p>';
    return;
  }
  let html = '';
  signals.slice(0, 8).forEach(s => {
    const dir = s.direction === 'buy' ? '▲ Покупка' : '▼ Продажа';
    const cls = s.direction === 'buy' ? 'green' : 'red';
    html += `<div class="signal-card">
      <span class="${cls}">${s.symbol}</span>
      <span>${dir}</span>
      <span style="color:var(--muted)">${new Date(s.timestamp).toLocaleString('ru-RU')}</span>
    </div>`;
  });
  el.innerHTML = html;
}

// ── Equity chart ───────────────────────────────────
function renderEquity(equity) {
  const ctx = document.getElementById('equity-chart');
  if (!ctx) return;
  if (equity.length < 2) {
    ctx.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)">Нет данных для графика</div>';
    return;
  }
  // Simple SVG line chart
  const w = ctx.clientWidth || 600, h = ctx.clientHeight || 250;
  const min = Math.min(...equity), max = Math.max(...equity);
  const range = max - min || 1;
  const pts = equity.map((v, i) => {
    const x = (i / (equity.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  ctx.innerHTML = `<svg width="100%" height="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="#3b82f6" stroke-width="2"/>
  </svg>`;
}

// ── Init ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshAll();
  // Auto-refresh every 30s
  setInterval(refreshAll, 30000);
});
