const $ = s => document.querySelector(s);
const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => n == null ? '—' : (n * 100).toFixed(4) + '%';
const mean = a => a.length ? a.reduce((s, x) => s + x.score, 0) / a.length : null;
const date = v => v ? new Date(v).toLocaleString('zh-TW', {timeZone:'Asia/Taipei',hour12:false}) : '尚未更新';
let state;
async function api(path, body) {
  const res = await fetch('/api/' + path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw Error(data.error || res.statusText);
  return data;
}
async function action(path, body = {}) {
  try {
    const result = await api(path, body);
    $('#notice').textContent = path === 'submit-preview'
      ? `檢查通過，未上傳：${result.model} / ${result.tradingDate} / ${result.stocks} 支股票。${result.wouldSubmit ? '檔案可提交。' : '相同預測已有紀錄：' + result.previousStatus}`
      : path === 'config' ? '設定已儲存' : result.queued ? '工作已排入佇列' : '這項工作已在佇列或執行中';
    await refresh();
  }
  catch(e) { $('#notice').textContent = e.message; }
}
function stat(label, value, detail) { return `<article class="stat"><small>${esc(label)}</small><strong>${esc(value)}</strong><p>${esc(detail)}</p></article>`; }
function render(s) {
  state = s;
  const c = s.competition || {}, p = c.phase2 || {}, ranks = c.ranks || {}, submissions = c.submissions || {};
  $('#busy').textContent = s.busy ? '執行中 · ' + s.busy : '待命中';
  $('#sync').textContent = '同步 05:10 / 22:10 · 自動提交 08:30（台北） · 最近檢查 ' + date(s.lastCheck);
  $('#stats').innerHTML = stat('Phase 2 場次', p.tradingDate || '未取得場次', p.deadline ? (p.isOpen ? '截止 ' : '目前未開放 · 截止 ') + date(p.deadline) : '開放交易日當天 09:00（台北時間）截止') + stat('剩餘提交額度', p.remaining ?? '待確認', '選用模型：' + (s.selected || '尚未選擇')) + stat('Phase 1 排名', ranks.phase1?.rank ?? '—', '參賽隊伍 ' + (ranks.phase1?.totalTeams ?? '—')) + stat('Phase 2 模擬總排名', ranks.phase2?.mockFinalRank ?? '—', '累積平均名次 ' + (ranks.phase2?.cumulativeAverageRank ?? '—'));
  $('#selected').textContent = s.selected ? '提交模型 / ' + s.selected : '尚未選擇提交模型';
  $('#auto').checked = s.autoSubmit;
  const submissionPending = s.pending.some(key => key === 'submit:' || key === 'scheduled-submit:');
  $('#submit').disabled = !s.selected || submissionPending;
  const lastResult = s.events.find(event => event.submission);
  const receipt = submissions.phase2?.latest;
  const submissionStatus = submissionPending
    ? (s.busy === 'submit:' || s.busy === 'scheduled-submit:' ? '正在檢查或提交，請稍候。' : '提交已排隊。')
    : lastResult ? `${date(lastResult.at)} · ${lastResult.message}`
      : receipt ? `官方最新收件 ${date(receipt.submittedAt)} · ${receipt.tradingDate} · ${receipt.status}`
        : '提交結果會顯示在這裡。';
  if ($('#submit-status').textContent !== submissionStatus) $('#submit-status').textContent = submissionStatus;
  $('#model-count').textContent = s.availableModels.length + ' models';
  $('#models').innerHTML = s.availableModels.length ? s.availableModels.map(name => {
    const runs = s.runs.filter(r => r.model === name), last = runs.at(-1), status = s.models[name] || {};
    // Latest forecast completed before each deadline, without replacing it with a retrospective run.
    const days = new Map();
    for (const r of runs) for (const score of r.scores) if (score.prospective) days.set(score.date, score);
    const live = [...days.values()];
    return `<article class="model ${s.selected === name ? 'selected' : ''}"><h3>${esc(name)} ${s.selected === name ? '✓' : ''}</h3><small>${esc(status.status || '尚未執行')}</small>${status.error ? `<p class="error">${esc(status.error)}</p>` : ''}<div class="metrics"><div class="metric"><small>即時日均分數 · ${live.length} 日</small><strong>${fmt(mean(live))}</strong></div><div class="metric"><small>最新預測歷史回算</small><strong>${fmt(mean((last?.scores || []).filter(x => !x.prospective)))}</strong></div></div><div class="actions"><button data-action="train" data-model="${esc(name)}" ${s.busy ? 'disabled' : ''}>重新訓練＋預測</button><button class="secondary" data-action="predict" data-model="${esc(name)}" ${s.busy ? 'disabled' : ''}>產生預測</button><button class="secondary" data-select="${esc(name)}">${s.selected === name ? '已選為提交模型' : '選為提交模型'}</button></div></article>`;
  }).join('') : '<div class="empty">尚未找到模型。請在 repo 的 <code>model/&lt;model_name&gt;/usage.txt</code> 填入兩行指令：第一行預測，第二行訓練。</div>';
  $('#runs').innerHTML = [...s.runs].reverse().slice(0, 100).map(r => `<tr><td>${esc(r.model)}<small>${esc(date(r.createdAt))}</small></td><td>${esc(r.status)}${r.error ? `<small class="error">${esc(r.error)}</small>` : ''}</td><td>${fmt(mean(r.scores.filter(x => x.prospective)))}</td><td>${fmt(mean(r.scores.filter(x => !x.prospective)))}</td><td>${r.scores.length} / ${r.dates?.length ?? '—'}</td><td><a href="/api/runs/${encodeURIComponent(r.id)}/log" target="_blank" rel="noopener">Log</a>${r.status === 'done' ? `<a href="/api/runs/${encodeURIComponent(r.id)}/csv" target="_blank" rel="noopener">CSV</a>` : ''}</td></tr>`).join('') || '<tr><td colspan="6">尚無預測紀錄</td></tr>';
  const extra = [];
  if (submissions.phase2?.latest) extra.push({at:submissions.phase2.latest.submittedAt,message:'官方最新提交：' + submissions.phase2.latest.file + ' · ' + (submissions.phase2.latest.score == null ? 'Pending' : fmt(submissions.phase2.latest.score))});
  for (const item of [...s.submissions].reverse().slice(0, 5)) extra.push({at:item.at,message:`提交 ${item.model}：${item.status}${item.error ? ' / ' + item.error : ''}`});
  $('#events').innerHTML = [...extra, ...s.events].slice(0, 35).map(e => `<li><time>${esc(date(e.at))}</time><span>${esc(e.message)}</span></li>`).join('') || '<li>啟動後會自動同步比賽資訊與資料。</li>';
}
async function refresh() { try { render(await api('status')); } catch(e) { $('#notice').textContent = '連線失敗：' + e.message; } }
document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.action) action(b.dataset.action, b.dataset.model ? {model:b.dataset.model} : {});
  if (b.dataset.select) action('config', {selected:b.dataset.select});
});
$('#auto').addEventListener('change', e => action('config', {autoSubmit:e.target.checked}));
refresh(); setInterval(refresh, 5000);
