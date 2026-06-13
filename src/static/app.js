'use strict';

/* ── App state ── */

let currentChatId  = null;
let currentClassId = null;
let currentTopicId = null;
let sessions  = [];
let topics    = [];
let myClasses = [];

/* ── Sessions ── */

async function fetchSessions() {
  if (!currentClassId) return;
  try {
    const r = await fetch(`/classes/${currentClassId}/chats/me`, {headers:getHeaders()});
    if (!r.ok) return;
    sessions = await r.json();
    renderSessions();
  } catch {}
}

async function deleteSessionRemote(id) {
  try { await fetch(`/chats/${id}`, {method:'DELETE', headers:getHeaders()}); } catch {}
  sessions = sessions.filter(s => s.id !== id);
  if (currentChatId === id) { currentChatId = null; clearOutput(); }
  renderSessions();
}

function renderSessions() {
  const list  = document.getElementById('sessions-list');
  const noEl  = document.getElementById('no-sessions');
  list.querySelectorAll('.session-item').forEach(e => e.remove());
  if (!sessions.length) { noEl.style.display='block'; return; }
  noEl.style.display = 'none';
  sessions.forEach(s => {
    const el = mk('div','session-item'+(s.id===currentChatId?' active':''));
    el.dataset.id = s.id;
    el.innerHTML = `
      <svg class="si-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
      </svg>
      <div class="si-info">
        <div class="si-title">${esc(s.title||'Untitled')}</div>
        <div class="si-time">${relTime(s.updated_at||s.created_at)}</div>
      </div>
      <button class="si-del" title="Delete chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>`;
    el.querySelector('.si-del').addEventListener('click', async e => {
      e.stopPropagation();
      if (!await showConfirm('Delete this chat?', 'Delete')) return;
      await deleteSessionRemote(s.id);
    });
    el.addEventListener('click', () => { closeMobileSidebar(); loadSessionById(s.id); });
    list.appendChild(el);
  });
}

async function loadSessionById(id) {
  try {
    const r = await fetch(`/chats/${id}`, {headers:getHeaders()});
    if (!r.ok) return;
    const chat = await r.json();
    currentChatId  = id;
    currentTopicId = chat.topic_id || null;
    renderTopicChip();
    renderSessions();
    renderHistory(chat.messages || []);
  } catch {}
}

function renderHistory(messages) {
  const out = document.getElementById('output');
  out.innerHTML = '';
  if (!messages.length) { appendEmpty(); return; }
  const inner = mk('div','output-inner');
  out.appendChild(inner);
  let i = 0;
  while (i < messages.length) {
    const um = messages[i];
    if (um.role !== 'user') { i++; continue; }
    const am = messages[i+1] && messages[i+1].role==='assistant' ? messages[i+1] : null;
    const block = mk('div','response'); block.style.animation='none';
    const qb = mk('div','q-bubble'); qb.textContent=um.content; block.appendChild(qb);
    if (am) {
      const sources = am.sources_json ? JSON.parse(am.sources_json) : [];
      const at = mk('div','answer-text prose');
      if (am.content) {
        const prefix = 'hr-' + am.id;
        const srcMap = {};
        sources.forEach((s, i) => { srcMap[`${s.source.trim()}|${String(s.page).trim()}`] = i; });
        let html = marked.parse(am.content, {breaks:false, gfm:true});
        html = html.replace(/\[Source:\s*([^\],]+?)(?:,\s*|\s*\|\s*)Page:\s*(\d+)\]/g, (_, src, pg) => {
          const key = `${src.trim()}|${pg.trim()}`;
          const i   = srcMap[key];
          if (i === undefined) return '';
          return `<sup><button class="cite-btn" onclick="document.getElementById('${prefix}-${i}').scrollIntoView({behavior:'smooth',block:'nearest'})">${i+1}</button></sup>`;
        });
        at.innerHTML = html;
      }
      block.appendChild(at);
      if (sources.length) {
        const prefix = 'hr-' + am.id;
        const sw = mk('div');
        sw.append(lbl('Sources'));
        const sl = mk('div','src-list');
        sources.forEach((s, idx) => {
          const item = mk('div','sl-item'); item.id = `${prefix}-${idx}`;
          const name = s.source.replace(/_topic[0-9a-f]{8}$/i, '');
          item.innerHTML = `<span class="sl-num">${idx+1}</span><div class="sl-info"><div class="sl-name" title="${esc(s.source)}">${esc(name)}</div><div class="sl-meta">Page ${s.page} · ${Math.round(s.relevance*100)}% relevance</div></div>`;
          sl.appendChild(item);
        });
        sw.appendChild(sl); block.appendChild(sw);
      }
      const existing = am.feedback_rating ? {rating:am.feedback_rating, comment:am.feedback_comment} : null;
      appendFeedbackBar(block, currentChatId, am.id, existing);
    }
    inner.appendChild(block);
    i += am ? 2 : 1;
  }
  out.scrollTop = out.scrollHeight;
}

/* ── Connection ── */

const connDot = document.getElementById('conn-dot');
function setConnState(s) {
  connDot.className = 'conn-dot ' + s;
  connDot.title = {checking:'Connecting…', online:'Server connected', offline:'Server offline'}[s];
}

/* ── Offline ── */

const offlineScreen = document.getElementById('offline-screen');
function showOffline() { offlineScreen.classList.add('open'); setConnState('offline'); }
function hideOffline() { offlineScreen.classList.remove('open'); }

document.getElementById('retry-btn').addEventListener('click', async () => { hideOffline(); await init(); });

/* ── User bar ── */

function updateUserBar(username, role) {
  const bar = document.getElementById('sb-user');
  if (!username) { bar.style.display='none'; return; }
  bar.style.display = 'flex';
  document.getElementById('user-avatar').textContent    = username[0].toUpperCase();
  document.getElementById('user-name-lbl').textContent  = username;
  const badge = document.getElementById('role-badge');
  badge.textContent = role; badge.className = 'role-badge ' + role;
  const isStaff = role==='teacher' || role==='supradmin';
  document.getElementById('sb-admin').style.display            = isStaff ? 'block' : 'none';
  document.getElementById('manage-classes-btn').style.display  = role==='supradmin' ? 'flex' : 'none';
  document.getElementById('manage-users-btn').style.display    = role==='supradmin' ? 'flex' : 'none';
  const esBody = document.getElementById('es-body');
  if (esBody) esBody.textContent = isStaff
    ? 'Open the Admin panel, create a topic, upload materials, then start a conversation.'
    : 'Pick a topic from the composer below, then ask your first question.';
}

document.getElementById('signout-btn').addEventListener('click', () => {
  clearToken(); clearUser();
  localStorage.removeItem(LS_CLASS);
  currentChatId=null; currentClassId=null; currentTopicId=null;
  sessions=[]; topics=[]; myClasses=[];
  document.getElementById('sb-user').style.display='none';
  document.getElementById('class-pill').style.display='none';
  showLoginModal();
});

/* ── Class picker ── */

function showClassPicker(required=false) {
  const modal = document.getElementById('class-picker-modal');
  const list  = document.getElementById('class-picker-list');
  const sub   = document.getElementById('class-picker-sub');
  sub.textContent = required ? 'Select the class you want to work in.' : 'Switch to a different class.';
  list.innerHTML  = '';
  if (!myClasses.length) {
    const p = mk('p','no-classes-msg');
    p.textContent = 'You are not enrolled in any classes yet. Ask your teacher or admin to add you.';
    list.appendChild(p);
    modal.classList.add('open'); return;
  }
  myClasses.forEach(cls => {
    const item = mk('div','class-item'+(cls.id===currentClassId?' active':''));
    item.innerHTML = `<div class="ci-name">${esc(cls.name)}</div>${cls.description?`<div class="ci-desc">${esc(cls.description)}</div>`:''}`;
    item.addEventListener('click', () => { modal.classList.remove('open'); selectClass(cls.id); });
    list.appendChild(item);
  });
  modal.classList.add('open');
}

document.getElementById('class-picker-modal').addEventListener('click', e => {
  if (e.target===e.currentTarget) e.currentTarget.classList.remove('open');
});
document.getElementById('class-pill').addEventListener('click', () => showClassPicker(false));

async function selectClass(classId) {
  currentClassId = classId;
  localStorage.setItem(LS_CLASS, classId);
  const cls  = myClasses.find(c => c.id===classId);
  const pill = document.getElementById('class-pill');
  document.getElementById('class-pill-name').textContent = cls?.name || '—';
  pill.style.display = 'flex';
  const mhClass = document.getElementById('mh-class-name');
  if (mhClass) mhClass.textContent = cls?.name || '';
  currentChatId=null; currentTopicId=null;
  clearOutput();
  await Promise.all([fetchTopics(), fetchSessions()]);
}

async function fetchClasses() {
  try {
    const r = await fetch('/classes/me', {headers:getHeaders()});
    if (!r.ok) return [];
    return await r.json();
  } catch { return []; }
}

/* ── Auth ── */

const loginModal = document.getElementById('login-modal');

function showLoginModal() {
  document.getElementById('login-user-input').value = '';
  document.getElementById('login-pw-input').value   = '';
  document.getElementById('login-error').style.display   = 'none';
  document.getElementById('login-panel').style.display   = '';
  document.getElementById('register-panel').style.display = 'none';
  loginModal.classList.add('open');
  setTimeout(() => document.getElementById('login-user-input').focus(), 80);
}

function hideLoginModal() { loginModal.classList.remove('open'); }

function showRegisterPanel() {
  document.getElementById('login-panel').style.display   = 'none';
  document.getElementById('register-panel').style.display = '';
  document.getElementById('reg-user-input').value  = '';
  document.getElementById('reg-pw-input').value    = '';
  document.getElementById('reg-error').style.display = 'none';
  loadRegisterClasses();
  setTimeout(() => document.getElementById('reg-user-input').focus(), 80);
}

async function loadRegisterClasses() {
  const sel = document.getElementById('reg-class-select');
  sel.innerHTML = '<option value="">Loading…</option>';
  try {
    const r       = await fetch('/auth/classes');
    const classes = r.ok ? await r.json() : [];
    sel.innerHTML = classes.length
      ? '<option value="">Choose your class…</option>' + classes.map(c=>`<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('')
      : '<option value="">No classes available yet</option>';
  } catch { sel.innerHTML='<option value="">Could not load classes</option>'; }
}

document.getElementById('show-register-btn').addEventListener('click', showRegisterPanel);
document.getElementById('show-login-btn').addEventListener('click', () => {
  document.getElementById('login-panel').style.display    = '';
  document.getElementById('register-panel').style.display = 'none';
  setTimeout(() => document.getElementById('login-user-input').focus(), 80);
});
document.getElementById('reg-submit-btn').addEventListener('click', doRegister);
['reg-user-input','reg-pw-input'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if(e.key==='Enter') doRegister(); })
);

async function doRegister() {
  const username = document.getElementById('reg-user-input').value.trim();
  const password = document.getElementById('reg-pw-input').value;
  const classId  = document.getElementById('reg-class-select').value;
  const err = document.getElementById('reg-error');
  err.style.display = 'none';
  if (!username)       { err.textContent='Username is required.';                      err.style.display='block'; return; }
  if (password.length<4) { err.textContent='Password must be at least 4 characters.'; err.style.display='block'; return; }
  if (!classId)        { err.textContent='Please choose a class.';                     err.style.display='block'; return; }
  try {
    const r = await fetch('/auth/register', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username, password, class_id:classId}),
    });
    if (!r.ok) { const d=await r.json(); err.textContent=d.detail||'Registration failed.'; err.style.display='block'; return; }
    const data = await r.json();
    storeToken(data.token);
    storeUser({id:data.user.id, username:data.user.username, role:data.user.role});
    hideLoginModal();
    updateUserBar(data.user.username, data.user.role);
    await afterLogin();
  } catch { err.textContent='Cannot reach server — is it running?'; err.style.display='block'; }
}

document.getElementById('login-submit-btn').addEventListener('click', doLogin);
['login-user-input','login-pw-input'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if(e.key==='Enter') doLogin(); })
);

async function doLogin() {
  const username = document.getElementById('login-user-input').value.trim();
  const pw       = document.getElementById('login-pw-input').value;
  const err = document.getElementById('login-error');
  err.style.display = 'none';
  if (!username) { err.textContent='Username is required.'; err.style.display='block'; return; }
  try {
    const r = await fetch('/auth/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username, password:pw}),
    });
    if (!r.ok) {
      err.textContent='Incorrect username or password.';
      err.style.display='block';
      document.getElementById('login-pw-input').select(); return;
    }
    const data = await r.json();
    storeToken(data.token);
    storeUser({id:data.user.id, username:data.user.username, role:data.user.role});
    hideLoginModal();
    updateUserBar(data.user.username, data.user.role);
    await afterLogin();
  } catch { err.textContent='Cannot reach server — is it running?'; err.style.display='block'; }
}

async function afterLogin() {
  myClasses = await fetchClasses();
  const savedId = localStorage.getItem(LS_CLASS);
  const saved   = myClasses.find(c => c.id===savedId);
  const role    = getUser().role;
  if (saved)                                await selectClass(saved.id);
  else if (myClasses.length===1)            await selectClass(myClasses[0].id);
  else if (myClasses.length>1&&role==='student') showClassPicker(true);
  else if (myClasses.length>0)              await selectClass(myClasses[0].id);
}

/* ── Topics chip/dropdown ── */

async function fetchTopics() {
  if (!currentClassId) return;
  try {
    const r = await fetch(`/classes/${currentClassId}/topics`, {headers:getHeaders()});
    if (!r.ok) return;
    topics = await r.json();
    renderTopicChip();
  } catch {}
}

function renderTopicChip() {
  const chip = document.getElementById('topic-chip');
  if (!topics.length) { chip.style.display='none'; return; }
  chip.style.display = 'flex';
  if (currentTopicId) {
    const t = topics.find(t => t.id===currentTopicId);
    chip.classList.add('active');
    chip.querySelector('.tc-label').textContent = t ? t.name : 'Unknown topic';
  } else {
    chip.classList.remove('active');
    chip.querySelector('.tc-label').textContent = 'All material';
  }
}

const topicDrop = document.getElementById('topic-dropdown');
document.getElementById('topic-chip').addEventListener('click', e => {
  e.stopPropagation();
  if (topicDrop.style.display==='none') { renderTopicDropdown(); topicDrop.style.display='block'; }
  else topicDrop.style.display='none';
});
document.addEventListener('click', () => { topicDrop.style.display='none'; });
topicDrop.addEventListener('click', e => e.stopPropagation());

function renderTopicDropdown() {
  topicDrop.innerHTML = '';
  const allItem = mk('div','td-item'+(!currentTopicId?' selected':''));
  allItem.innerHTML = '<span class="td-item-name">All material</span>';
  allItem.addEventListener('click', () => { setTopic(null); topicDrop.style.display='none'; });
  topicDrop.appendChild(allItem);
  if (topics.length) {
    topicDrop.appendChild(mk('div','td-sep'));
    topics.forEach(t => {
      const item = mk('div','td-item'+(currentTopicId===t.id?' selected':''));
      item.innerHTML = `<span class="td-item-name">${esc(t.name)}</span><span class="td-item-count">${t.doc_count||0} docs</span>`;
      item.addEventListener('click', () => { setTopic(t.id); topicDrop.style.display='none'; });
      topicDrop.appendChild(item);
    });
  }
}

function setTopic(topicId) { currentTopicId=topicId; renderTopicChip(); }

/* ── Mobile sidebar ── */

function openMobileSidebar() {
  document.querySelector('.sidebar').classList.add('mobile-open');
  document.getElementById('sidebar-overlay').classList.add('open');
}
function closeMobileSidebar() {
  document.querySelector('.sidebar').classList.remove('mobile-open');
  document.getElementById('sidebar-overlay').classList.remove('open');
}
document.getElementById('mobile-menu-btn').addEventListener('click', openMobileSidebar);
document.getElementById('sidebar-overlay').addEventListener('click', closeMobileSidebar);

/* ── New chat / output helpers ── */

document.getElementById('new-chat-btn').addEventListener('click', () => {
  closeMobileSidebar();
  currentChatId=null; currentTopicId=null;
  renderTopicChip(); renderSessions(); clearOutput();
});

function clearOutput() { document.getElementById('output').innerHTML=''; appendEmpty(); }

function appendEmpty() {
  const out  = document.getElementById('output');
  const es   = mk('div',''); es.id='empty-state';
  const role = getUser().role || 'student';
  const bodyText = role==='student'
    ? 'Pick a topic from the composer below, then ask your first question.'
    : 'Open the Admin panel, create a topic, upload materials, then start a conversation.';
  es.innerHTML=`
    <div class="es-orb"><div class="es-center">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
      </svg>
    </div></div>
    <p class="es-title">What do you want to learn?</p>
    <p class="es-body" id="es-body">${bodyText}</p>`;
  out.appendChild(es);
}

/* ── Admin panel nav ── */

function showAdminPanel(name) {
  document.getElementById('panel-title').textContent = {topics:'Topics',classes:'Classes',chats:'Browse Chats',feedback:'Feedback'}[name] || name;
  document.getElementById('chat-view').style.display  = 'none';
  document.getElementById('admin-panel').style.display = 'flex';
  if (name==='topics')        loadTopicsPanel();
  else if (name==='classes')  loadClassesPanel();
  else if (name==='chats')    loadChatsPanel();
  else if (name==='feedback') loadFeedbackPanel();
}

function hideAdminPanel() {
  document.getElementById('admin-panel').style.display = 'none';
  document.getElementById('chat-view').style.display   = 'flex';
}

document.getElementById('admin-toggle').addEventListener('click', () => document.getElementById('sb-admin').classList.toggle('open'));
document.getElementById('manage-topics-btn').addEventListener('click', () => { closeMobileSidebar(); showAdminPanel('topics'); });
document.getElementById('manage-classes-btn').addEventListener('click', () => { closeMobileSidebar(); showAdminPanel('classes'); });
document.getElementById('manage-users-btn').addEventListener('click', () => { closeMobileSidebar(); showUsersModal(); });
document.getElementById('browse-chats-btn').addEventListener('click', () => { closeMobileSidebar(); showAdminPanel('chats'); });
document.getElementById('view-feedback-btn').addEventListener('click', () => { closeMobileSidebar(); showAdminPanel('feedback'); });
document.getElementById('panel-back-btn').addEventListener('click', hideAdminPanel);

/* ── Reveal buttons ── */

document.querySelectorAll('.reveal-btn').forEach(btn => {
  const EYE     = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const EYE_OFF = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
  btn.innerHTML = EYE;
  btn.addEventListener('click', () => {
    const inp  = document.getElementById(btn.dataset.for);
    const show = inp.type==='password'; inp.type=show?'text':'password'; btn.innerHTML=show?EYE_OFF:EYE;
  });
});

/* ── Init ── */

async function init() {
  setConnState('checking');
  try {
    const r = await fetch('/health', {signal:AbortSignal.timeout(4000)});
    if (!r.ok) { showOffline(); return; }
    setConnState('online');
    const ar = await fetch('/auth/me', {headers:getHeaders()});
    if (!ar.ok) { showLoginModal(); return; }
    const user = await ar.json();
    storeUser({id:user.id, username:user.username, role:user.role});
    updateUserBar(user.username, user.role);
    myClasses = await fetchClasses();
    const savedId = localStorage.getItem(LS_CLASS);
    const saved   = myClasses.find(c => c.id===savedId);
    if (saved)                                     await selectClass(saved.id);
    else if (myClasses.length===1)                 await selectClass(myClasses[0].id);
    else if (myClasses.length>1&&user.role==='student') showClassPicker(true);
    else if (myClasses.length>0)                   await selectClass(myClasses[0].id);
  } catch { showOffline(); }
}

/* ── Boot ── */

(function() {
  const mac  = /Mac|iPhone|iPad|iPod/.test(navigator.platform);
  const hint = document.getElementById('send-hint');
  if (hint) hint.innerHTML = mac ? '<kbd>⌘</kbd><kbd>↵</kbd> to send' : '<kbd>Ctrl</kbd><kbd>↵</kbd> to send';
})();

init();
