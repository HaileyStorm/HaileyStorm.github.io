import { EXPECTED, OFFICIAL, findPublished, publishedView, summaryRows, chartSVG } from './model.mjs';
const button = document.querySelector('#check');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const download = document.querySelector('#download');
const certificateLink = document.querySelector('#certificate-link');
let objectURL = null;

async function officialJSON(relative) {
  const url = new URL(relative, OFFICIAL).href;
  const response = await fetch(url, { method:'GET', mode:'cors', credentials:'omit', cache:'no-store', redirect:'error', referrerPolicy:'no-referrer' });
  if (!response.ok || response.url !== url) throw Error('The official source could not be read.');
  return response.json();
}

async function checkPublication() {
  if (button.disabled) return;
  button.disabled = true;
  result.hidden = true; download.hidden = true; certificateLink.hidden = true;
  if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
  download.removeAttribute('href'); document.querySelector('#chart').removeAttribute('src');
  status.textContent = 'Reading the official catalogue once…';
  try {
    const catalogue = await officialJSON('catalogue.json');
    findPublished(catalogue);
    const certificate = await officialJSON(`certificates/${EXPECTED.id}.json`);
    const view = publishedView(catalogue, certificate);
    const metrics = document.querySelector('#metrics');
    metrics.replaceChildren();
    for (const [key,value] of summaryRows(view)) {
      const dt=document.createElement('dt'), dd=document.createElement('dd');
      dt.textContent=key; dd.textContent=value; metrics.append(dt,dd);
    }
    const svg = chartSVG(view);
    objectURL = URL.createObjectURL(new Blob([svg], {type:'image/svg+xml;charset=utf-8'}));
    const image = document.querySelector('#chart');
    image.src = objectURL;
    image.alt = `Published Cycle092 tracker history. ${view.classification}. Signed changes and rank are given in the preceding comparison.`;
    download.href = objectURL;
    document.querySelector('#exact').textContent = JSON.stringify({
      result_id:EXPECTED.id, source_commit:EXPECTED.head, official_catalogue:OFFICIAL+'catalogue.json',
      certificate_url:OFFICIAL+`certificates/${EXPECTED.id}.json`, report_url:certificate.report_url,
      report_sha256:certificate.report_sha256, kernels:certificate.kernels,
      previous_submission:view.chosen.priorSubmission.id, signed_delta_c_previous_submission:view.chosen.submissionDelta,
      previous_frontier:view.chosen.priorFrontier.id, signed_delta_c_previous_frontier:view.chosen.frontierDelta,
      announcement_at:view.timeline, verified_at:view.verifiedAt, catalogue_as_of:view.catalogueAsOf,
      rank_excluding_baseline:view.rank, N_excluding_baseline:view.N,
      rank_including_baseline:view.rankIncludingBaseline, N_including_baseline:view.NIncludingBaseline,
    },null,2);
    const link=document.createElement('a');
    link.href=OFFICIAL+`certificates/${EXPECTED.id}.json`; link.textContent='Matching official three-kernel verification certificate';
    certificateLink.replaceChildren(link); certificateLink.hidden=false;
    result.hidden=false; download.hidden=false;
    status.textContent='Published result and verification certificate match this exact submission head. The chart is ready to view or download.';
  } catch (error) {
    status.textContent = 'Chart unavailable. '+(error instanceof TypeError ? 'The official data could not be read in this browser.' : error.message)+
      ' Available after matching catalogue publication; use the button for another check when you choose.';
  } finally { button.disabled=false; }
}
button.addEventListener('click', checkPublication);
checkPublication(); // One initial read, then explicit user clicks only.
