// Public metadata validation and exact arithmetic; no network or DOM effects.
// Rational history is the verbatim upstream Apache-2.0 module beside this file.
import { rational, cmp, sub, fraction, scientific, recordHistory } from './rational.mjs';
export const OFFICIAL = 'https://gtimg.github.io/QuasiRiemannTracker/';
export const EXPECTED = Object.freeze({ id: 'hailey-collet-cycle092', pr: 144,
  repository: 'GTimG/QuasiRiemannTracker', head: '50a933d719e4adf2775fabe9774a53000dfea522',
  theta: '436666397/500000000', sourcePath:'submissions/2026-11-20-hailey-collet-cycle092/src/Candidate/Main.lean' });
const KERNELS = ['Lean default', 'nanoda', 'con-ron'];
const fail = (message) => { throw Error(message); };
const require = (test, message) => { if (!test) fail(message); };
const instant = (value) => {
  require(typeof value === 'string' && Number.isFinite(Date.parse(value)), 'Official source dates are incomplete.');
  return new Date(value).toISOString();
};
const ranked = (record) => rational(record.theta?.numerator, record.theta?.denominator);
const ZERO = rational('0', '1');
const ONE = rational('1', '1');
const eligible = (record) => ['accepted', 'framework-verified', 'verified'].includes(record.status);
export const signed = (value) => (cmp(value, ZERO) > 0 ? '+' : '') + scientific(value);

export function findPublished(catalogue, expected = EXPECTED) {
  require(catalogue?.schema_version === 1 && Array.isArray(catalogue.records), 'Official catalogue format is unavailable.');
  require(new Set(catalogue.records.map(r => r.id)).size === catalogue.records.length, 'Official catalogue has ambiguous result identities.');
  const matches = catalogue.records.filter(r => r.id === expected.id);
  require(matches.length === 1, 'This exact result is not yet listed in the official catalogue.');
  const record = matches[0];
  require(record.status === 'framework-verified', 'The official entry has no completed framework verification.');
  require(record.repository === expected.repository && record.source_commit === expected.head && record.pr === expected.pr,
    'The official publication does not match the specified submission and final head.');
  require(fraction(ranked(record)) === expected.theta && !record.exact_bound,
    'The published statement does not match this exact rational bound.');
  require(record.certificate_url === `certificates/${expected.id}.json` || record.certificate_url === `${OFFICIAL}certificates/${expected.id}.json`,
    'The official entry has no supported public verification certificate.');
  require(record.solution_url === `https://github.com/${expected.repository}/blob/${expected.head}/${expected.sourcePath}`,
    'The published source link does not identify the expected submitted source.');
  instant(record.first_verified_at);
  return record;
}

export function publishedView(catalogue, certificate, expected = EXPECTED) {
  const record = findPublished(catalogue, expected);
  require(certificate?.schema_version === 1 && certificate.result_id === expected.id &&
    certificate.source_repository === expected.repository && certificate.source_commit === expected.head &&
    certificate.theta_exact === expected.theta, 'The public certificate does not match this exact result and head.');
  require(certificate.status === undefined || ['PASS','accepted','success'].includes(certificate.status),
    'The certificate has an unsupported completion status.');
  require(JSON.stringify(certificate.kernels) === JSON.stringify(KERNELS), 'The public certificate does not identify all three required kernels.');
  require(instant(certificate.verified_at) === instant(record.first_verified_at), 'Published verification dates disagree.');
  require(/^[a-f0-9]{64}$/.test(certificate.report_sha256 ?? ''), 'The public certificate is missing its verification-report digest.');
  const escaped = expected.repository.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  require(new RegExp(`^https://github\\.com/${escaped}/(?:actions/runs/[0-9]+(?:/attempts/[0-9]+)?|blob/[a-f0-9]{40}/[^\\s?#]+)$`).test(certificate.report_url ?? ''),
    'The public certificate has no supported canonical verification-report link.');
  const visible = catalogue.records.filter(eligible).map(r => ({ ...r, theta: ranked(r),
    actual_verified_at: instant(r.first_verified_at), first_verified_at: instant(r.timeline_at || r.first_verified_at) }));
  require(visible.length <= 10000 && visible.every(r => typeof r.id === 'string'), 'Official history is unavailable.');
  require(visible.every(r => cmp(r.theta, ZERO) > 0 && cmp(r.theta, ONE) <= 0), 'Official ranked history is outside the supported range.');
  const history = recordHistory(visible);
  let frontier = null, previous = null;
  const rows = history.map(r => {
    const row = { ...r, timeline: r.first_verified_at, priorSubmission: previous, priorFrontier: frontier,
      submissionDelta: previous ? sub(previous.theta, r.theta) : null,
      frontierDelta: frontier ? sub(frontier.theta, r.theta) : null };
    previous = r;
    if (r.is_record) frontier = r;
    return row;
  });
  const chosen = rows.find(r => r.id === expected.id);
  require(chosen && chosen.priorSubmission && chosen.priorFrontier, 'Official predecessor history is incomplete.');
  const through = rows.slice(0, rows.indexOf(chosen)+1);
  const events = through.filter(r => r.is_record && r.frontierDelta && cmp(r.frontierDelta, ZERO) > 0);
  const excluding = events.filter(r => r.id !== 'openai-baseline');
  const rank = cohort => chosen.is_record ? 1+cohort.filter(r => cmp(r.frontierDelta, chosen.frontierDelta) > 0).length : null;
  return { expected, certificate, catalogueAsOf: catalogue.as_of || 'unspecified', rows, chosen,
    c: sub(ONE, chosen.theta), theta: chosen.theta,
    rank: rank(excluding), N: excluding.length, rankIncludingBaseline: rank(events), NIncludingBaseline: events.length,
    verifiedAt: certificate.verified_at, timeline: chosen.timeline, entryCount: rows.length,
    classification: chosen.is_record ? 'Strict rational-coordinate record' : cmp(chosen.frontierDelta, ZERO) === 0 ? 'Tied rational coordinate' : 'Non-record rational coordinate' };
}

export function summaryRows(view) {
  const r = view.chosen;
  const label = r => `${r.id}${r.pr ? ` (PR${r.pr})` : ''}`;
  return [ ['Published bound', `c = ${fraction(view.c)}; θ = ${fraction(view.theta)}`],
    ['Classification', view.classification], ['Previous submission', `${label(r.priorSubmission)}; signed Δc ${signed(r.submissionDelta)}`],
    ['Previous strict frontier', `${label(r.priorFrontier)}; signed Δc ${signed(r.frontierDelta)}`],
    ['Positive-jump rank', view.rank ? `${view.rank} of ${view.N} excluding baseline; ${view.rankIncludingBaseline} of ${view.NIncludingBaseline} including baseline` : 'No positive record-jump rank'],
    ['Announcement date plotted', view.timeline], ['Verification date', view.verifiedAt],
    ['Catalogue snapshot', `${view.catalogueAsOf}; ${view.entryCount} eligible published entries`],
    ['Rank cohort', 'Positive strict rational-coordinate jumps through this announcement; equal sizes share a rank'] ];
}

const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
export function chartSVG(view, fixtureLabel = '') {
  require(view.expected.id === EXPECTED.id || fixtureLabel === 'FIXTURE ONLY', 'Synthetic charts require an explicit fixture label.');
  const selected = view.chosen;
  const title = `${fixtureLabel ? fixtureLabel+' · Synthetic fixture record' : 'Cycle092'} · published tracker history`;
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="1180" viewBox="0 0 1320 1180" role="img" aria-labelledby="title desc">`,
    `<title id="title">${esc(title)}</title><desc id="desc">${esc(summaryRows(view).map(([k,v]) => k+': '+v).join('. '))}</desc>`,
    `<metadata>${esc(JSON.stringify({ fixture:fixtureLabel || null, expected:view.expected,
      catalogue:OFFICIAL+'catalogue.json', certificate:OFFICIAL+`certificates/${view.expected.id}.json`,
      report_url:view.certificate.report_url, report_sha256:view.certificate.report_sha256,
      signed_delta_c_previous_submission:fraction(selected.submissionDelta),
      signed_delta_c_previous_frontier:fraction(selected.frontierDelta),
      rank_excluding_baseline:view.rank, N_excluding_baseline:view.N,
      rank_including_baseline:view.rankIncludingBaseline, N_including_baseline:view.NIncludingBaseline,
      announcement_at:view.timeline, verified_at:view.verifiedAt, catalogue_as_of:view.catalogueAsOf }))}</metadata>`,
    '<rect width="100%" height="100%" fill="#fbfcfc"/>'];
  const text = (x,y,t,size=17,color='#13202a',anchor='start') => parts.push(`<text x="${x}" y="${y}" font-family="DejaVu Sans, sans-serif" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(t)}</text>`);
  const line = (x,y,u,v,color='#dce3e7',width=1) => parts.push(`<line x1="${x}" y1="${y}" x2="${u}" y2="${v}" stroke="${color}" stroke-width="${width}"/>`);
  text(65,48,title,27);
  text(65,82,`Official catalogue + matching three-kernel certificate · verified ${view.verifiedAt}`,16,'#52616c');
  text(65,114,`c = ${fraction(view.c)} · θ = ${fraction(view.theta)} · ${view.entryCount} published entries`,18);
  const recent = view.rows.filter(r => Date.parse(r.timeline) >= Date.parse('2026-10-01T00:00:00Z'));
  require(recent.length > 0, 'No supported modern plotting history.');
  const value = r => Number(r.theta.numerator)/Number(r.theta.denominator);
  const day = 86400000;
  const end = Math.max(...recent.map(r => Date.parse(r.timeline)))+day;
  function panel(start, top, height, detail) {
    const active = recent.filter(r => Date.parse(r.timeline) >= start);
    require(active.length > 0, 'No supported detail plotting history.');
    let low = Math.min(...active.map(value)), high = Math.max(...active.map(value));
    const pad = Math.max((high-low)*.15, detail ? .0000001 : .00001);
    low -= pad; high += pad;
    const X = at => 140+(at-start)/(end-start)*1090;
    const Y = v => top+(high-v)/(high-low)*height;
    for (let i=0; i<5; i++) {
      const v = low+(high-low)*i/4, y=Y(v), t=start+(end-start)*i/4;
      line(140,y,1230,y);text(126,y+5,v.toFixed(detail?8:7),14,'#52616c','end');
      text(X(t),top+height+28,new Date(t).toISOString().slice(5,10),14,'#52616c','middle');
    }
    let front = null;
    for (const r of view.rows) {
      const t = Date.parse(r.timeline), v=value(r);
      if (t < start) { if (r.is_record && v>=low && v<=high) front=[140,Y(v)]; continue; }
      if (t > end || v<low || v>high) continue;
      const x=X(t),y=Y(v), color=r.id===selected.id?'#ad761e':'#1d5d74';
      if (r.is_record) { if (front) {line(front[0],front[1],x,front[1],'#1d5d74',2);line(x,front[1],x,y,'#1d5d74',2);} front=[x,y]; }
      parts.push(`<circle cx="${x}" cy="${y}" r="${r.id===selected.id?8:4}" stroke="${color}" fill="${r.is_record?color:'#fbfcfc'}" stroke-width="2"/>`);
    }
  }
  text(65,150,'θ (smaller is stronger)',16,'#52616c');
  panel(Math.min(...recent.map(r=>Date.parse(r.timeline)))-day,175,350,false);
  text(65,605,'Recent detail · same rational coordinates',19);
  panel(Date.parse(selected.timeline)-14*day,635,190,true);
  text(65,905,`Previous submission: ${selected.priorSubmission.id} · signed Δc ${signed(selected.submissionDelta)}`,16);
  text(65,940,`Previous strict frontier: ${selected.priorFrontier.id} · signed Δc ${signed(selected.frontierDelta)}`,16);
  text(65,980,view.rank ? `Positive jump rank ${view.rank} / ${view.N} excluding baseline; ${view.rankIncludingBaseline} / ${view.NIncludingBaseline} including baseline` : 'No positive strict record jump; signed changes remain shown above.',17);
  text(65,1020,`Announcement ${selected.timeline} · catalogue as of ${view.catalogueAsOf}`,15,'#52616c');
  text(65,1055,'Filled: strict rational record · hollow: other published entry · gold: this exact published result',15,'#52616c');
  text(65,1090,'Exact algebraic improvements may share a rational coordinate. Exact signed fractions accompany this chart.',15,'#52616c');
  text(65,1130,`${fixtureLabel ? fixtureLabel+' · ' : ''}${view.expected.id} · source ${view.expected.head}`,14,'#52616c');
  parts.push('</svg>');
  return parts.join('\n');
}
