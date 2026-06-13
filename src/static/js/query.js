'use strict';

/* ── Query area DOM refs ── */

const output        = document.getElementById('output');
const questionInput = document.getElementById('question-input');
const askBtn        = document.getElementById('ask-btn');
const askLabel      = document.getElementById('ask-label');

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + 'px';
});
questionInput.addEventListener('keydown', e => { if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();submitQuery();} });
askBtn.addEventListener('click', submitQuery);

/* ── Pipeline ── */

const STEPS   = ['hypothesis','sources','answering'];
const SLABELS = {hypothesis:'Thinking', sources:'Searching', answering:'Writing'};

function makePipeline() {
  const wrap=mk('div','pipeline'), els={};
  STEPS.forEach((s,i) => {
    const step=mk('span','pip-step'), dot=mk('span','pip-dot');
    step.append(dot, document.createTextNode(SLABELS[s]));
    wrap.appendChild(step); els[s]=step;
    if (i<STEPS.length-1) { const c=mk('span','pip-connector'); c.dataset.step=s; wrap.appendChild(c); }
  });
  return {wrap, els};
}

function setPipActive(els, wrap, step) {
  const idx = STEPS.indexOf(step);
  STEPS.forEach((s,i) => { els[s].className=i<idx?'pip-step done':i===idx?'pip-step active':'pip-step'; });
  wrap.querySelectorAll('.pip-connector').forEach((c,i) => c.classList.toggle('done', i<idx));
}

function setPipDone(els, wrap) {
  STEPS.forEach(s => { els[s].className='pip-step done'; });
  wrap.querySelectorAll('.pip-connector').forEach(c => c.classList.add('done'));
}

function getInner() {
  let inner = output.querySelector('.output-inner');
  if (!inner) { inner=mk('div','output-inner'); output.appendChild(inner); }
  return inner;
}

function scrollEnd() { requestAnimationFrame(() => { output.scrollTop=output.scrollHeight; }); }

function setLocked(v) {
  askBtn.disabled=v; questionInput.disabled=v;
  if (v) {
    askLabel.textContent='Working';
    const icon=document.getElementById('ask-icon');
    if (icon) icon.outerHTML=`<div class="spinner" id="ask-icon"></div>`;
  } else {
    const sp=document.getElementById('ask-icon');
    if (sp) sp.outerHTML=`<svg id="ask-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`;
    askLabel.textContent='Ask';
  }
}

/* ── Source helpers ── */

function cleanSourceName(raw) {
  return raw.replace(/_topic[0-9a-f]{8}$/i, '');
}

function renderSourceList(sources, container, prefix) {
  sources.forEach((s, i) => {
    const item = mk('div', 'sl-item');
    item.id = `${prefix}-${i}`;
    const name = cleanSourceName(s.source);
    const pct  = Math.round(s.relevance * 100);
    item.innerHTML =
      `<span class="sl-num">${i+1}</span>` +
      `<div class="sl-info">` +
        `<div class="sl-name" title="${esc(s.source)}">${esc(name)}</div>` +
        `<div class="sl-meta">Page ${s.page} · ${pct}% relevance</div>` +
      `</div>`;
    container.appendChild(item);
  });
}

function injectCiteButtons(html, sources, prefix) {
  const srcMap = {};
  sources.forEach((s, i) => {
    const key = `${s.source.trim()}|${String(s.page).trim()}`;
    if (srcMap[key] === undefined) srcMap[key] = i;
  });
  return html.replace(/\[Source:\s*([^\],]+?)(?:,\s*|\s*\|\s*)Page:\s*(\d+)\]/g, (_, src, pg) => {
    const key = `${src.trim()}|${pg.trim()}`;
    const i   = srcMap[key];
    if (i === undefined) return '';
    return `<sup><button class="cite-btn" onclick="document.getElementById('${prefix}-${i}').scrollIntoView({behavior:'smooth',block:'nearest'})">${i+1}</button></sup>`;
  });
}

/* ── Submit query ── */

async function submitQuery() {
  const question = questionInput.value.trim();
  if (!question || askBtn.disabled) return;
  if (!currentClassId) return;

  if (!currentChatId) {
    try {
      const cr = await fetch(`/classes/${currentClassId}/chats`, {
        method:'POST', headers:getHeaders({'Content-Type':'application/json'}),
        body:JSON.stringify({topic_id:currentTopicId||null}),
      });
      if (!cr.ok) return;
      currentChatId = (await cr.json()).id;
    } catch { return; }
  }

  const es = document.getElementById('empty-state');
  if (es) es.remove();
  setLocked(true);
  questionInput.value=''; questionInput.style.height='auto';

  const inner     = getInner();
  const block     = mk('div','response');
  const qb        = mk('div','q-bubble'); qb.textContent=question;
  const {wrap:pipWrap, els:pipEls} = makePipeline();
  setPipActive(pipEls, pipWrap, 'hypothesis');
  const thinkWrap = mk('div','thinking-wrap');
  [72,58,40].forEach((w,i) => { const l=mk('div','shimmer-line'); l.style.cssText=`width:${w}%;animation-delay:${i*0.18}s`; thinkWrap.appendChild(l); });
  const ansSec    = mk('div'); ansSec.style.display='none';
  const ansText   = mk('div','answer-text prose');
  const cursor    = mk('span','cursor');
  ansSec.appendChild(ansText);
  const srcSec    = mk('div'); srcSec.style.display='none';
  const srcList   = mk('div','src-list');
  srcSec.append(lbl('Sources'), srcList);
  block.append(qb, pipWrap, thinkWrap, ansSec, srcSec);
  inner.appendChild(block);
  scrollEnd();

  const prefix = 'sr-' + Date.now();
  let collAnswer='', collSources=[], asstMsgId=null;

  try {
    const res = await fetch(`/chats/${currentChatId}/query/stream`, {
      method:'POST',
      headers:getHeaders({'Content-Type':'application/json'}),
      body:JSON.stringify({question}),
    });

    if (!res.ok) {
      thinkWrap.remove(); ansSec.style.display='block';
      ansText.textContent = await friendlyError(res);
      setPipDone(pipEls,pipWrap); return;
    }

    const reader=res.body.getReader(), decoder=new TextDecoder();
    let buf='';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream:true});
      const parts = buf.split('\n\n'); buf=parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        let evt; try { evt=JSON.parse(part.slice(6)); } catch { continue; }

        if (evt.type==='hyde') {
          setPipActive(pipEls,pipWrap,'sources');

        } else if (evt.type==='sources') {
          collSources=evt.sources;
          thinkWrap.remove();
          ansSec.style.display='block'; ansText.appendChild(cursor);
          setPipActive(pipEls,pipWrap,'answering');
          scrollEnd();

        } else if (evt.type==='delta') {
          collAnswer += evt.text;
          ansText.insertBefore(document.createTextNode(evt.text), cursor);
          scrollEnd();

        } else if (evt.type==='error') {
          if (thinkWrap.parentNode) thinkWrap.remove();
          cursor.remove();
          ansSec.style.display='block';
          const errPrefix = ansText.textContent ? '\n\n' : '';
          ansText.appendChild(document.createTextNode(`${errPrefix}⚠ ${evt.message}`));
          setPipDone(pipEls,pipWrap);

        } else if (evt.type==='done') {
          cursor.remove();
          // Render answer as markdown and inject inline cite buttons
          if (collAnswer.trim()) {
            let html = marked.parse(collAnswer, {breaks:false, gfm:true});
            html = injectCiteButtons(html, collSources, prefix);
            ansText.innerHTML = html;
          }
          // Render source list below the answer
          if (collSources.length) {
            renderSourceList(collSources, srcList, prefix);
            srcSec.style.display='block';
          }
          setPipDone(pipEls,pipWrap);
          asstMsgId = evt.message_id;
          if (asstMsgId) appendFeedbackBar(block, currentChatId, asstMsgId, null);
          await fetchSessions();
          scrollEnd();
        }
      }
    }
  } catch {
    thinkWrap.remove(); cursor.remove(); ansSec.style.display='block';
    ansText.appendChild(document.createTextNode(ansText.textContent ? '\n\n⚠ Connection lost.' : 'Network error — is the server running?'));
    setPipDone(pipEls,pipWrap);
  } finally {
    setLocked(false);
    questionInput.focus();
  }
}

/* ── Feedback bar ── */

function appendFeedbackBar(block, chatId, msgId, existing) {
  const bar        = mk('div','feedback-bar');
  const upBtn      = mk('button','fb-btn up');   upBtn.innerHTML='👍'; upBtn.title='Helpful';
  const downBtn    = mk('button','fb-btn down'); downBtn.innerHTML='👎'; downBtn.title='Not helpful';
  const statusEl   = mk('span','fb-status');
  const commentRow = mk('div','fb-comment-row');
  const commentIn  = mk('input','fb-comment-input');
  commentIn.placeholder='Optional comment…'; commentIn.type='text';
  const submitBtn  = mk('button','fb-submit'); submitBtn.textContent='Submit';
  commentRow.append(commentIn, submitBtn);
  let pendingRating = null;
  if (existing) {
    if (existing.rating==='up')   upBtn.classList.add('active');
    if (existing.rating==='down') downBtn.classList.add('active');
    if (existing.comment) statusEl.textContent=`"${trunc(existing.comment,60)}"`;
  }
  function selectRating(r) { pendingRating=r; upBtn.classList.toggle('active',r==='up'); downBtn.classList.toggle('active',r==='down'); commentRow.classList.add('visible'); commentIn.focus(); }
  upBtn.addEventListener('click',   ()=>selectRating('up'));
  downBtn.addEventListener('click', ()=>selectRating('down'));
  submitBtn.addEventListener('click', async () => {
    if (!pendingRating) return;
    const comment = commentIn.value.trim();
    submitBtn.disabled=true; statusEl.textContent='Saving…';
    try {
      const r = await fetch(`/chats/${chatId}/messages/${msgId}/feedback`, {
        method:'POST', headers:getHeaders({'Content-Type':'application/json'}),
        body:JSON.stringify({rating:pendingRating, comment:comment||null}),
      });
      if (!r.ok) { statusEl.textContent="Couldn't save — try again."; submitBtn.disabled=false; return; }
      commentRow.classList.remove('visible');
      statusEl.textContent='Thanks for the feedback';
    } catch { statusEl.textContent='Network error.'; submitBtn.disabled=false; }
  });
  bar.append(upBtn, downBtn, statusEl, commentRow);
  block.appendChild(bar);
}
