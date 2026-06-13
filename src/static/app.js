'use strict';

/* ── Confirm dialog ── */
function showConfirm(msg, okLabel='Confirm') {
  return new Promise(resolve => {
    const [title, body=''] = msg.split('\n\n');
    document.getElementById('confirm-title').textContent = title;
    document.getElementById('confirm-body').textContent  = body;
    document.getElementById('confirm-ok').textContent    = okLabel;
    const overlay = document.getElementById('confirm-modal');
    overlay.classList.add('open');
    const ok  = document.getElementById('confirm-ok');
    const can = document.getElementById('confirm-cancel');
    function done(v) {
      overlay.classList.remove('open');
      ok.removeEventListener('click', onOk);
      can.removeEventListener('click', onCancel);
      overlay.removeEventListener('click', onBg);
      resolve(v);
    }
    function onOk()     { done(true);  }
    function onCancel() { done(false); }
    function onBg(e)    { if (e.target===overlay) done(false); }
    ok.addEventListener('click', onOk);
    can.addEventListener('click', onCancel);
    overlay.addEventListener('click', onBg);
  });
}

/* ── Storage helpers ── */

const LS_TOKEN   = 'rag_auth_token';

const LS_USER    = 'rag_user_info';

const LS_CLASS   = 'rag_class_id';

const LS_TOPK    = 'rag_topk';

function getToken()    { return localStorage.getItem(LS_TOKEN) || ''; }

function storeToken(t) { localStorage.setItem(LS_TOKEN, t); }

function clearToken()  { localStorage.removeItem(LS_TOKEN); }

function getUser()     { try { return JSON.parse(localStorage.getItem(LS_USER))||{}; } catch { return {}; } }

function storeUser(u)  { localStorage.setItem(LS_USER, JSON.stringify(u)); }

function clearUser()   { localStorage.removeItem(LS_USER); }



function getHeaders(extra={}) {

  const h = {...extra};

  const tok = getToken();

  if (tok) h['Authorization'] = 'Bearer ' + tok;

  return h;

}



/* ── App state ── */

let currentChatId  = null;   // UUID from server

let currentClassId = null;

let currentTopicId = null;

let sessions  = [];  // cached from GET /classes/{id}/chats/me

let topics    = [];  // cached from GET /classes/{id}/topics

let myClasses = [];  // cached from GET /classes/me



/* ─────────────────────── Sessions ─────────────────────── */



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



function relTime(iso) {

  const s = Math.floor((Date.now()-new Date(iso))/1000);

  if (s<60)  return 'just now';

  const m=Math.floor(s/60);  if (m<60)  return m+'m ago';

  const h=Math.floor(m/60);  if (h<24)  return h+'h ago';

  const d=Math.floor(h/24);  if (d<7)   return d+'d ago';

  return new Date(iso).toLocaleDateString();

}



function renderSessions() {

  const list=document.getElementById('sessions-list');

  const noEl=document.getElementById('no-sessions');

  list.querySelectorAll('.session-item').forEach(e=>e.remove());

  if (!sessions.length) { noEl.style.display='block'; return; }

  noEl.style.display='none';

  sessions.forEach(s=>{

    const el=mk('div','session-item'+(s.id===currentChatId?' active':''));

    el.dataset.id=s.id;

    el.innerHTML=`

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

    el.querySelector('.si-del').addEventListener('click', async e=>{

      e.stopPropagation();

      if (!await showConfirm('Delete this chat?', 'Delete')) return;

      await deleteSessionRemote(s.id);

    });

    el.addEventListener('click', ()=>loadSessionById(s.id));

    list.appendChild(el);

  });

}



async function loadSessionById(id) {

  try {

    const r=await fetch(`/chats/${id}`, {headers:getHeaders()});

    if (!r.ok) return;

    const chat=await r.json();

    currentChatId  = id;

    currentTopicId = chat.topic_id||null;

    renderTopicChip();

    renderSessions();

    renderHistory(chat.messages||[]);

  } catch {}

}



function renderHistory(messages) {

  const out=document.getElementById('output');

  out.innerHTML='';

  if (!messages.length) { appendEmpty(); return; }

  const inner=mk('div','output-inner');

  out.appendChild(inner);

  let i=0;

  while (i<messages.length) {

    const um=messages[i];

    if (um.role!=='user') { i++; continue; }

    const am=messages[i+1]&&messages[i+1].role==='assistant'?messages[i+1]:null;

    const block=mk('div','response'); block.style.animation='none';

    const qb=mk('div','q-bubble'); qb.textContent=um.content; block.appendChild(qb);

    if (am) {

      const sources=am.sources_json?JSON.parse(am.sources_json):[];

      if (sources.length) {

        const sw=mk('div');

        sw.append(lbl('Sources used'));

        const sg=mk('div','source-grid');

        sources.forEach((s,idx)=>{

          const pct=Math.round(s.relevance*100);

          const c=mk('div','source-card'); c.style.animation='none';

          c.innerHTML=`<div class="sc-top"><span class="sc-rank">Source ${idx+1}</span><span class="sc-pct">${pct}%</span></div>

            <div class="sc-name" title="${esc(s.source)}">${esc(s.source)}</div>

            <div class="sc-page">Page ${s.page}</div>

            <div class="rel-track"><div class="rel-fill" style="width:${pct}%;transition:none"></div></div>`;

          sg.appendChild(c);

        });

        sw.appendChild(sg); block.appendChild(sw);

      }

      const at=mk('div','answer-text'); at.textContent=am.content;

      block.appendChild(at);

      const existing=am.feedback_rating?{rating:am.feedback_rating, comment:am.feedback_comment}:null;

      appendFeedbackBar(block, currentChatId, am.id, existing);

    }

    inner.appendChild(block);

    i += am?2:1;

  }

  out.scrollTop=out.scrollHeight;

}



/* ─────────────────────── Connection ─────────────────────── */



const connDot=document.getElementById('conn-dot');

function setConnState(s) {

  connDot.className='conn-dot '+s;

  connDot.title={checking:'Connecting…',online:'Server connected',offline:'Server offline'}[s];

}



/* ─────────────────────── Offline ─────────────────────── */



const offlineScreen=document.getElementById('offline-screen');

function showOffline() { offlineScreen.classList.add('open'); setConnState('offline'); }

function hideOffline() { offlineScreen.classList.remove('open'); }

document.getElementById('retry-btn').addEventListener('click', async ()=>{ hideOffline(); await init(); });



/* ─────────────────────── User bar ─────────────────────── */



function updateUserBar(username, role) {

  const bar=document.getElementById('sb-user');

  if (!username) { bar.style.display='none'; return; }

  bar.style.display='flex';

  document.getElementById('user-avatar').textContent=username[0].toUpperCase();

  document.getElementById('user-name-lbl').textContent=username;

  const badge=document.getElementById('role-badge');

  badge.textContent=role; badge.className='role-badge '+role;

  const isStaff = role==='teacher'||role==='supradmin';

  document.getElementById('sb-admin').style.display = isStaff?'block':'none';

  document.getElementById('manage-classes-btn').style.display = role==='supradmin'?'flex':'none';

  document.getElementById('manage-users-btn').style.display   = role==='supradmin'?'flex':'none';

  const esBody=document.getElementById('es-body');

  if (esBody) esBody.textContent = isStaff

    ? 'Open the Admin panel, create a topic, upload materials, then start a conversation.'

    : 'Pick a topic from the composer below, then ask your first question.';

}



document.getElementById('signout-btn').addEventListener('click', ()=>{

  clearToken(); clearUser();

  localStorage.removeItem(LS_CLASS);

  currentChatId=null; currentClassId=null; currentTopicId=null;

  sessions=[]; topics=[]; myClasses=[];

  document.getElementById('sb-user').style.display='none';

  document.getElementById('class-pill').style.display='none';

  showLoginModal();

});



/* ─────────────────────── Class picker ─────────────────────── */



function showClassPicker(required=false) {

  const modal=document.getElementById('class-picker-modal');

  const list=document.getElementById('class-picker-list');

  const sub=document.getElementById('class-picker-sub');

  sub.textContent=required?'Select the class you want to work in.':'Switch to a different class.';

  list.innerHTML='';

  if (!myClasses.length) {

    const p=mk('p','no-classes-msg');

    p.textContent='You are not enrolled in any classes yet. Ask your teacher or admin to add you.';

    list.appendChild(p);

    modal.classList.add('open'); return;

  }

  myClasses.forEach(cls=>{

    const item=mk('div','class-item'+(cls.id===currentClassId?' active':''));

    item.innerHTML=`<div class="ci-name">${esc(cls.name)}</div>${cls.description?`<div class="ci-desc">${esc(cls.description)}</div>`:''}`;

    item.addEventListener('click', ()=>{ modal.classList.remove('open'); selectClass(cls.id); });

    list.appendChild(item);

  });

  modal.classList.add('open');

}

document.getElementById('class-picker-modal').addEventListener('click', e=>{

  if (e.target===e.currentTarget) e.currentTarget.classList.remove('open');

});

document.getElementById('class-pill').addEventListener('click', ()=>showClassPicker(false));



async function selectClass(classId) {

  currentClassId=classId;

  localStorage.setItem(LS_CLASS, classId);

  const cls=myClasses.find(c=>c.id===classId);

  const pill=document.getElementById('class-pill');

  document.getElementById('class-pill-name').textContent=cls?.name||'—';

  pill.style.display='flex';

  currentChatId=null; currentTopicId=null;

  clearOutput();

  await Promise.all([fetchTopics(), fetchSessions()]);

}



/* ─────────────────────── Login modal ─────────────────────── */



const loginModal=document.getElementById('login-modal');

function showLoginModal() {

  document.getElementById('login-user-input').value='';

  document.getElementById('login-pw-input').value='';

  document.getElementById('login-error').style.display='none';

  document.getElementById('login-panel').style.display='';

  document.getElementById('register-panel').style.display='none';

  loginModal.classList.add('open');

  setTimeout(()=>document.getElementById('login-user-input').focus(),80);

}

function hideLoginModal() { loginModal.classList.remove('open'); }



function showRegisterPanel() {

  document.getElementById('login-panel').style.display='none';

  document.getElementById('register-panel').style.display='';

  document.getElementById('reg-user-input').value='';

  document.getElementById('reg-pw-input').value='';

  document.getElementById('reg-error').style.display='none';

  loadRegisterClasses();

  setTimeout(()=>document.getElementById('reg-user-input').focus(),80);

}



async function loadRegisterClasses() {

  const sel=document.getElementById('reg-class-select');

  sel.innerHTML='<option value="">Loading…</option>';

  try {

    const r=await fetch('/auth/classes');

    const classes=r.ok?await r.json():[];

    sel.innerHTML=classes.length

      ?'<option value="">Choose your class…</option>'+classes.map(c=>`<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('')

      :'<option value="">No classes available yet</option>';

  } catch { sel.innerHTML='<option value="">Could not load classes</option>'; }

}



document.getElementById('show-register-btn').addEventListener('click', showRegisterPanel);

document.getElementById('show-login-btn').addEventListener('click', ()=>{

  document.getElementById('login-panel').style.display='';

  document.getElementById('register-panel').style.display='none';

  setTimeout(()=>document.getElementById('login-user-input').focus(),80);

});

document.getElementById('reg-submit-btn').addEventListener('click', doRegister);

['reg-user-input','reg-pw-input'].forEach(id=>

  document.getElementById(id).addEventListener('keydown', e=>{ if(e.key==='Enter') doRegister(); })

);



async function doRegister() {

  const username=document.getElementById('reg-user-input').value.trim();

  const password=document.getElementById('reg-pw-input').value;

  const classId =document.getElementById('reg-class-select').value;

  const err=document.getElementById('reg-error');

  err.style.display='none';

  if (!username) { err.textContent='Username is required.'; err.style.display='block'; return; }

  if (password.length<4) { err.textContent='Password must be at least 4 characters.'; err.style.display='block'; return; }

  if (!classId) { err.textContent='Please choose a class.'; err.style.display='block'; return; }

  try {

    const r=await fetch('/auth/register',{

      method:'POST', headers:{'Content-Type':'application/json'},

      body:JSON.stringify({username, password, class_id:classId}),

    });

    if (!r.ok) {

      const d=await r.json();

      err.textContent=d.detail||'Registration failed.'; err.style.display='block'; return;

    }

    const data=await r.json();

    storeToken(data.token);

    storeUser({id:data.user.id, username:data.user.username, role:data.user.role});

    hideLoginModal();

    updateUserBar(data.user.username, data.user.role);

    await afterLogin();

  } catch {

    err.textContent='Cannot reach server — is it running?'; err.style.display='block';

  }

}



document.getElementById('login-submit-btn').addEventListener('click', doLogin);

['login-user-input','login-pw-input'].forEach(id=>

  document.getElementById(id).addEventListener('keydown', e=>{ if(e.key==='Enter') doLogin(); })

);



async function doLogin() {

  const username=document.getElementById('login-user-input').value.trim();

  const pw=document.getElementById('login-pw-input').value;

  const err=document.getElementById('login-error');

  err.style.display='none';

  if (!username) { err.textContent='Username is required.'; err.style.display='block'; return; }

  try {

    const r=await fetch('/auth/login',{

      method:'POST', headers:{'Content-Type':'application/json'},

      body:JSON.stringify({username, password:pw}),

    });

    if (!r.ok) {

      err.textContent='Incorrect username or password.';

      err.style.display='block';

      document.getElementById('login-pw-input').select(); return;

    }

    const data=await r.json();

    storeToken(data.token);

    storeUser({id:data.user.id, username:data.user.username, role:data.user.role});

    hideLoginModal();

    updateUserBar(data.user.username, data.user.role);

    await afterLogin();

  } catch {

    err.textContent='Cannot reach server — is it running?';

    err.style.display='block';

  }

}



async function afterLogin() {

  myClasses=await fetchClasses();

  const savedId=localStorage.getItem(LS_CLASS);

  const saved=myClasses.find(c=>c.id===savedId);

  if (saved) await selectClass(saved.id);

  else if (myClasses.length===1) await selectClass(myClasses[0].id);

  else showClassPicker(true);

}



async function fetchClasses() {

  try {

    const r=await fetch('/classes/me',{headers:getHeaders()});

    if (!r.ok) return [];

    return await r.json();

  } catch { return []; }

}



/* ─────────────────────── Users modal ─────────────────────── */



const usersModal=document.getElementById('users-modal');

function showUsersModal() { loadUserList(); usersModal.classList.add('open'); }

function hideUsersModal() { usersModal.classList.remove('open'); }

document.getElementById('close-users-btn').addEventListener('click', hideUsersModal);



async function loadUserList() {

  const list=document.getElementById('user-list');

  list.innerHTML='<p style="font-size:12px;color:var(--text-muted);padding:6px 4px">Loading…</p>';

  try {

    const r=await fetch('/users',{headers:getHeaders()});

    if (!r.ok) { list.innerHTML='<p style="font-size:12px;color:var(--red);padding:6px 4px">Failed to load users.</p>'; return; }

    const users=await r.json();

    const me=getUser().id;

    list.innerHTML='';

    users.forEach(u=>{

      const el=mk('div','user-row');

      el.innerHTML=`

        <div class="ur-avatar">${esc(u.username[0].toUpperCase())}</div>

        <div class="ur-info">

          <span class="ur-name">${esc(u.username)}${u.id===me?' <span style="font-size:10px;color:var(--text-muted)">(you)</span>':''}</span>

          <span class="role-badge ${u.role}">${esc(u.role)}</span>

          ${!u.is_active?'<span style="font-size:10px;color:var(--red)">(inactive)</span>':''}

        </div>


        <button class="ur-toggle" title="${u.is_active?'Deactivate':'Activate'} user" ${u.id===me?'disabled':''}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            ${u.is_active
              ? '<circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/>'
              : '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>'}
          </svg>
        </button>

        <button class="ur-del" title="Permanently delete user" ${u.id===me?'disabled':''}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/>
          </svg>
        </button>`;

      if (u.id!==me) {

        el.querySelector('.ur-toggle').addEventListener('click', async ()=>{
          const path=u.is_active?'deactivate':'activate';
          await fetch(`/users/${u.id}/${path}`,{method:'PATCH',headers:getHeaders()});
          loadUserList();
        });

        el.querySelector('.ur-del').addEventListener('click', async ()=>{
          if (!await showConfirm(`Permanently delete "${u.username}"?

This cannot be undone. Their chats and enrollment history will be erased.`, 'Delete')) return;
          await fetch(`/users/${u.id}`,{method:'DELETE',headers:getHeaders()});
          loadUserList();
        });

      }

      list.appendChild(el);

    });

  } catch { list.innerHTML='<p style="font-size:12px;color:var(--red);padding:6px 4px">Error loading users.</p>'; }

}



document.getElementById('add-user-btn').addEventListener('click', addUser);

document.getElementById('new-user-password').addEventListener('keydown', e=>{ if(e.key==='Enter') addUser(); });



async function addUser() {

  const username=document.getElementById('new-user-username').value.trim();

  const password=document.getElementById('new-user-password').value;

  const role=document.getElementById('new-user-role').value;

  const err=document.getElementById('add-user-error');

  err.style.display='none';

  if (!username) { err.textContent='Username required.'; err.style.display='block'; return; }

  if (!password) { err.textContent='Password required.'; err.style.display='block'; return; }

  try {

    const r=await fetch('/users',{

      method:'POST', headers:getHeaders({'Content-Type':'application/json'}),

      body:JSON.stringify({username, password, role}),

    });

    if (!r.ok) {

      const d=await r.json();

      err.textContent=d.detail||'Failed to create user.'; err.style.display='block'; return;

    }

    document.getElementById('new-user-username').value='';

    document.getElementById('new-user-password').value='';

    loadUserList();

  } catch { err.textContent='Cannot reach server.'; err.style.display='block'; }

}



/* ─────────────────────── Topics ─────────────────────── */



async function fetchTopics() {

  if (!currentClassId) return;

  try {

    const r=await fetch(`/classes/${currentClassId}/topics`,{headers:getHeaders()});

    if (!r.ok) return;

    topics=await r.json();

    renderTopicChip();

  } catch {}

}



function renderTopicChip() {

  const chip=document.getElementById('topic-chip');

  if (!topics.length) { chip.style.display='none'; return; }

  chip.style.display='flex';

  if (currentTopicId) {

    const t=topics.find(t=>t.id===currentTopicId);

    chip.classList.add('active');

    chip.querySelector('.tc-label').textContent=t?t.name:'Unknown topic';

  } else {

    chip.classList.remove('active');

    chip.querySelector('.tc-label').textContent='All material';

  }

}



const topicDrop=document.getElementById('topic-dropdown');

document.getElementById('topic-chip').addEventListener('click', e=>{

  e.stopPropagation();

  if (topicDrop.style.display==='none') { renderTopicDropdown(); topicDrop.style.display='block'; }

  else topicDrop.style.display='none';

});

document.addEventListener('click', ()=>{ topicDrop.style.display='none'; });

topicDrop.addEventListener('click', e=>e.stopPropagation());



function renderTopicDropdown() {

  topicDrop.innerHTML='';

  const allItem=mk('div','td-item'+(!currentTopicId?' selected':''));

  allItem.innerHTML='<span class="td-item-name">All material</span>';

  allItem.addEventListener('click', ()=>{ setTopic(null); topicDrop.style.display='none'; });

  topicDrop.appendChild(allItem);

  if (topics.length) {

    topicDrop.appendChild(mk('div','td-sep'));

    topics.forEach(t=>{

      const item=mk('div','td-item'+(currentTopicId===t.id?' selected':''));

      item.innerHTML=`<span class="td-item-name">${esc(t.name)}</span><span class="td-item-count">${t.doc_count||0} docs</span>`;

      item.addEventListener('click', ()=>{ setTopic(t.id); topicDrop.style.display='none'; });

      topicDrop.appendChild(item);

    });

  }

}



function setTopic(topicId) { currentTopicId=topicId; renderTopicChip(); }



/* ─────────────────────── Topics panel ─────────────────────── */



async function loadTopicsPanel() {

  const body=document.getElementById('panel-body');

  body.innerHTML='';

  const createRow=mk('div','tp-create-row');

  const nameInput=document.createElement('input');

  nameInput.type='text'; nameInput.placeholder='New topic name…'; nameInput.autocomplete='off'; nameInput.spellcheck=false;

  const createBtn=document.createElement('button'); createBtn.textContent='Create topic';

  const cardsWrap=mk('div','tp-cards');

  createBtn.addEventListener('click', async ()=>{

    const name=nameInput.value.trim(); if (!name) return;

    const r=await fetch(`/classes/${currentClassId}/topics`,{

      method:'POST', headers:getHeaders({'Content-Type':'application/json'}),

      body:JSON.stringify({name}),

    });

    if (!r.ok) return;

    nameInput.value='';

    await fetchTopics(); renderTopicCards(cardsWrap);

  });

  nameInput.addEventListener('keydown', e=>{ if(e.key==='Enter') createBtn.click(); });

  createRow.append(nameInput, createBtn);

  body.append(createRow, cardsWrap);

  await fetchTopics();

  renderTopicCards(cardsWrap);

}



function renderTopicCards(container) {

  container.innerHTML='';

  if (!topics.length) {

    const empty=mk('p','tp-empty-files'); empty.textContent='No topics yet. Create one above.';

    container.appendChild(empty); return;

  }

  topics.forEach(topic=>{

    const card=mk('div','tp-card'); card.dataset.topicId=topic.id;

    const hdr=mk('div','tp-card-hdr');

    const nameEl=mk('span','tp-card-name'); nameEl.textContent=topic.name;

    const delBtn=mk('button','tp-card-del'); delBtn.title='Delete topic';

    delBtn.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/></svg>`;

    delBtn.addEventListener('click', async ()=>{

      if (!await showConfirm(`Delete topic "${topic.name}"?`, 'Delete')) return;

      await fetch(`/classes/${currentClassId}/topics/${topic.id}`,{method:'DELETE',headers:getHeaders()});

      await fetchTopics(); renderTopicCards(document.querySelector('.tp-cards'));

    });

    hdr.append(nameEl, delBtn); card.appendChild(hdr);

    const filesLbl=mk('p','tp-files-lbl'); filesLbl.textContent='Files'; card.appendChild(filesLbl);

    const filesWrap=mk('div','tp-files'); filesWrap.dataset.topicId=topic.id;

    card.appendChild(filesWrap);

    loadDocumentTags(filesWrap, topic.id);

    const drop=mk('div','tp-drop');

    const fileInput=document.createElement('input'); fileInput.type='file'; fileInput.multiple=true;

    fileInput.accept='.pdf,.txt,.md,.docx,.html,.htm,.epub,.odt';

    const dropLabel=mk('span','tp-drop-label'); dropLabel.textContent='Click to upload files — or drag them here';

    drop.append(fileInput, dropLabel);

    drop.addEventListener('click', e=>{ if(e.target!==fileInput) fileInput.click(); });

    drop.addEventListener('dragover', e=>{ e.preventDefault(); drop.classList.add('drag-over'); });

    drop.addEventListener('dragleave', ()=>drop.classList.remove('drag-over'));

    drop.addEventListener('drop', e=>{ e.preventDefault(); drop.classList.remove('drag-over'); handleFiles(Array.from(e.dataTransfer.files), topic.id, uploadingWrap, filesWrap); });

    fileInput.addEventListener('change', ()=>{ if (!fileInput.files.length) return; handleFiles(Array.from(fileInput.files), topic.id, uploadingWrap, filesWrap); fileInput.value=''; });

    card.appendChild(drop);

    const uploadingWrap=mk('div','tp-uploading'); card.appendChild(uploadingWrap);

    container.appendChild(card);

  });

}



async function loadDocumentTags(filesWrap, topicId) {

  filesWrap.innerHTML='';

  try {

    const r=await fetch(`/classes/${currentClassId}/topics/${topicId}/documents`,{headers:getHeaders()});

    if (!r.ok) { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'No files yet.'})); return; }

    const docs=await r.json();

    if (!docs.length) { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'No files yet.'})); return; }

    docs.forEach(doc=>{

      const tag=mk('span','tp-src-tag');

      tag.innerHTML=`${esc(doc.filename)}<button class="tp-src-del" title="Delete">×</button>`;

      tag.querySelector('.tp-src-del').addEventListener('click', async e=>{

        e.stopPropagation();

        if (!await showConfirm(`Delete "${doc.filename}"?`, 'Delete')) return;

        await fetch(`/classes/${currentClassId}/topics/${topicId}/documents/${doc.id}`,{method:'DELETE',headers:getHeaders()});

        loadDocumentTags(filesWrap, topicId);

      });

      filesWrap.appendChild(tag);

    });

  } catch { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'Error loading files.'})); }

}



function handleFiles(files, topicId, uploadingWrap, filesWrap) {

  files.forEach(file=>{

    const pill=mk('div','tp-upload-pill'); pill.textContent=`${file.name} — Uploading…`;

    uploadingWrap.appendChild(pill);

    ingestAndAssign(file, topicId, pill, filesWrap);

  });

}



async function ingestAndAssign(file, topicId, pill, filesWrap) {

  const form=new FormData(); form.append('file', file);

  try {

    const r=await fetch(`/classes/${currentClassId}/topics/${topicId}/documents`,{method:'POST',headers:getHeaders(),body:form});

    if (!r.ok) { pill.textContent=`${file.name} — Upload failed.`; pill.classList.add('error'); return; }

    pill.textContent=`✓ ${file.name}`; pill.classList.add('done');

    await fetchTopics();

    loadDocumentTags(filesWrap, topicId);

    setTimeout(()=>{ pill.style.transition='opacity 0.5s'; pill.style.opacity='0'; setTimeout(()=>pill.remove(),500); }, 3000);

  } catch { pill.textContent=`${file.name} — Upload failed.`; pill.classList.add('error'); }

}



/* ─────────────────────── Classes panel (supradmin) ─────────────────────── */



async function loadClassesPanel() {

  const body=document.getElementById('panel-body');

  body.innerHTML='';

  const createRow=mk('div','tp-create-row');

  const nameInput=document.createElement('input');

  nameInput.type='text'; nameInput.placeholder='New class name…'; nameInput.autocomplete='off';

  const createBtn=document.createElement('button'); createBtn.textContent='Create class';

  const cardsWrap=mk('div','tp-cards');

  createBtn.addEventListener('click', async ()=>{

    const name=nameInput.value.trim(); if (!name) return;

    const r=await fetch('/classes',{method:'POST',headers:getHeaders({'Content-Type':'application/json'}),body:JSON.stringify({name})});

    if (!r.ok) return;

    nameInput.value='';

    myClasses=await fetchClasses();

    renderClassCards(cardsWrap);

  });

  nameInput.addEventListener('keydown', e=>{ if(e.key==='Enter') createBtn.click(); });

  createRow.append(nameInput, createBtn);

  body.append(createRow, cardsWrap);

  // fetch all users for dropdowns

  await renderClassCards(cardsWrap);

}



async function renderClassCards(container) {

  container.innerHTML='<p style="font-size:12px;color:var(--text-muted);padding:6px">Loading…</p>';

  try {

    const [clsRes, usrRes]=await Promise.all([

      fetch('/classes',{headers:getHeaders()}),

      fetch('/users',{headers:getHeaders()}),

    ]);

    const allClasses=clsRes.ok?await clsRes.json():[];

    const allUsers=usrRes.ok?await usrRes.json():[];

    const teachers=allUsers.filter(u=>u.role==='teacher'&&u.is_active);

    const students=allUsers.filter(u=>u.role==='student'&&u.is_active);

    container.innerHTML='';

    if (!allClasses.length) { container.innerHTML='<p style="font-size:12px;color:var(--text-hint);font-style:italic">No classes yet.</p>'; return; }

    for (const cls of allClasses) {

      const card=mk('div','cls-card');

      const hdr=mk('div','cls-card-hdr');

      const nameEl=mk('span','cls-card-name'); nameEl.textContent=cls.name;

      if (!cls.is_active) { card.style.opacity='0.5'; card.style.pointerEvents='none'; const badge=mk('span','cls-inactive-badge'); badge.textContent='Inactive'; hdr.appendChild(badge); card.style.pointerEvents=''; }

      const toggleBtn=mk('button', cls.is_active?'cls-card-del':'cls-card-act');

      toggleBtn.title=cls.is_active?'Deactivate class':'Activate class';

      toggleBtn.innerHTML=cls.is_active

        ?`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`

        :`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`;

      toggleBtn.addEventListener('click', async ()=>{

        const action=cls.is_active?'deactivate':'activate';

        if (cls.is_active && !await showConfirm(`Deactivate class "${cls.name}"?

Students will lose access until it is re-activated.`, 'Deactivate')) return;

        await fetch(`/classes/${cls.id}/${action}`,{method:'PATCH',headers:getHeaders()});

        myClasses=await fetchClasses(); renderClassCards(container);

      });

      hdr.append(nameEl, toggleBtn); card.appendChild(hdr);

      // teachers section

      const teachersWrap=mk('div'); card.appendChild(teachersWrap);

      await renderMembersSection(teachersWrap, cls.id, 'teacher', 'Teachers', teachers);

      // students section

      const studentsWrap=mk('div'); card.appendChild(studentsWrap);

      await renderMembersSection(studentsWrap, cls.id, 'student', 'Students', students);

      container.appendChild(card);

    }

  } catch(e) { container.innerHTML=`<p style="font-size:12px;color:var(--red)">${esc(String(e))}</p>`; }

}



async function renderMembersSection(container, classId, memberType, label, candidates) {

  container.innerHTML='';

  const lbl=mk('p','cls-section-lbl'); lbl.textContent=label; container.appendChild(lbl);

  const membersDiv=mk('div','cls-members'); container.appendChild(membersDiv);

  const endpoint=memberType==='teacher'?'teachers':'students';

  try {

    const r=await fetch(`/classes/${classId}/${endpoint}`,{headers:getHeaders()});

    const members=r.ok?await r.json():[];

    if (members.length) {

      members.forEach(m=>{

        const tag=mk('span','cls-member-tag');

        tag.innerHTML=`${esc(m.username)}<button class="cls-member-del" title="Remove">×</button>`;

        tag.querySelector('.cls-member-del').addEventListener('click', async ()=>{

          await fetch(`/classes/${classId}/${endpoint}/${m.id}`,{method:'DELETE',headers:getHeaders()});

          renderMembersSection(container, classId, memberType, label, candidates);

        });

        membersDiv.appendChild(tag);

      });

    } else {

      membersDiv.appendChild(Object.assign(mk('span','cls-empty'),{textContent:`No ${label.toLowerCase()} yet.`}));

    }

    const addRow=mk('div','cls-add-row');

    const sel=document.createElement('select'); sel.innerHTML=`<option value="">Add ${memberType}…</option>`;

    candidates.forEach(u=>{ const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; sel.appendChild(o); });

    const addBtn=mk('button','cls-add-btn'); addBtn.textContent='Add';

    addBtn.addEventListener('click', async ()=>{

      if (!sel.value) return;

      await fetch(`/classes/${classId}/${endpoint}`,{method:'POST',headers:getHeaders({'Content-Type':'application/json'}),body:JSON.stringify({user_id:sel.value})});

      renderMembersSection(container, classId, memberType, label, candidates);

    });

    addRow.append(sel, addBtn); container.appendChild(addRow);

  } catch {}

}



/* ─────────────────────── Chats panel ─────────────────────── */



async function loadChatsPanel() {

  const body=document.getElementById('panel-body');

  body.innerHTML='';

  const filterRow=mk('div','panel-filter-row');

  const uSel=document.createElement('select'); uSel.id='ac-user-filter';

  uSel.innerHTML='<option value="">All users</option>';

  const tSel=document.createElement('select'); tSel.id='ac-topic-filter';

  tSel.innerHTML='<option value="">All topics</option>';

  topics.forEach(t=>{ const o=document.createElement('option'); o.value=t.id; o.textContent=t.name; tSel.appendChild(o); });

  const filterBtn=document.createElement('button'); filterBtn.textContent='Filter';

  filterRow.append(uSel, tSel, filterBtn);

  body.appendChild(filterRow);

  try {

    const r=await fetch(`/classes/${currentClassId}/students`,{headers:getHeaders()});

    if (r.ok) (await r.json()).forEach(u=>{ const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; uSel.appendChild(o); });

  } catch {}

  const listWrap=mk('div','panel-list'); body.appendChild(listWrap);

  filterBtn.addEventListener('click', ()=>renderChatsList(listWrap));

  await renderChatsList(listWrap);

}



async function renderChatsList(container) {

  container.innerHTML='<p class="admin-empty">Loading…</p>';

  const userId=document.getElementById('ac-user-filter')?.value||'';

  const topicId=document.getElementById('ac-topic-filter')?.value||'';

  const qs=new URLSearchParams();

  if (userId)  qs.set('user_id', userId);

  if (topicId) qs.set('topic_id', topicId);

  try {

    const r=await fetch(`/classes/${currentClassId}/chats?${qs}`,{headers:getHeaders()});

    if (!r.ok) { container.innerHTML='<p class="admin-empty">Failed to load.</p>'; return; }

    const chats=await r.json();

    container.innerHTML='';

    if (!chats.length) { container.innerHTML='<p class="admin-empty">No chats found.</p>'; return; }

    chats.forEach(c=>{

      const card=mk('div','admin-chat-card');

      const topicLabel=c.topic_id?(topics.find(t=>t.id===c.topic_id)?.name||c.topic_id):'All material';

      card.innerHTML=`

        <div class="acc-header">

          <span class="acc-title">${esc(c.title||'Untitled')}</span>

          <span class="acc-user-tag">${esc(c.username)}</span>

        </div>

        <div class="acc-meta">

          <span>${topicLabel}</span>

          <span>${new Date(c.updated_at).toLocaleDateString()}</span>

        </div>`;

      card.addEventListener('click', ()=>viewAdminChat(currentClassId, c.id, c.title, c.username));

      container.appendChild(card);

    });

  } catch { container.innerHTML='<p class="admin-empty">Error loading chats.</p>'; }

}



async function viewAdminChat(classId, chatId, title, username) {

  const body=document.getElementById('panel-body');

  body.innerHTML='';

  const backBtn=mk('button','admin-back-btn'); backBtn.innerHTML='← Back to list';

  backBtn.addEventListener('click', loadChatsPanel);

  const heading=mk('p','blk-label'); heading.textContent=`${username}: ${title||'Untitled'}`;

  body.append(backBtn, heading);

  try {

    const r=await fetch(`/classes/${classId}/chats/${chatId}/view`,{headers:getHeaders()});

    if (!r.ok) { body.appendChild(Object.assign(mk('p','admin-empty'),{textContent:'Failed to load chat.'})); return; }

    const chat=await r.json();

    const view=mk('div','admin-chat-view');

    const msgs=chat.messages||[];

    for (let i=0; i<msgs.length; i+=2) {

      const um=msgs[i]; const am=msgs[i+1]; if (!um) continue;

      const pair=mk('div','acv-pair');

      const qEl=mk('div','acv-q'); qEl.textContent=um.content; pair.appendChild(qEl);

      if (am) {

        const aEl=mk('div','acv-a'); aEl.textContent=am.content; pair.appendChild(aEl);

        if (am.feedback_rating) {

          const fb=mk('span',`acv-fb ${am.feedback_rating}`);

          fb.textContent=am.feedback_rating==='up'?'👍 Helpful':'👎 Not helpful';

          if (am.feedback_comment) fb.textContent+=` — "${am.feedback_comment}"`;

          pair.appendChild(fb);

        }

      }

      view.appendChild(pair);

    }

    body.appendChild(view);

  } catch { body.appendChild(Object.assign(mk('p','admin-empty'),{textContent:'Error loading chat.'})); }

}



/* ─────────────────────── Feedback panel ─────────────────────── */



async function loadFeedbackPanel() {

  const body=document.getElementById('panel-body');

  body.innerHTML='';

  const filterRow=mk('div','panel-filter-row');

  const uSel=document.createElement('select'); uSel.id='af-user-filter';

  uSel.innerHTML='<option value="">All users</option>';

  const rSel=document.createElement('select'); rSel.id='af-rating-filter';

  rSel.innerHTML='<option value="">All ratings</option><option value="up">👍 Positive</option><option value="down">👎 Negative</option>';

  const filterBtn=document.createElement('button'); filterBtn.textContent='Filter';

  filterRow.append(uSel, rSel, filterBtn);

  body.appendChild(filterRow);

  try {

    const r=await fetch(`/classes/${currentClassId}/students`,{headers:getHeaders()});

    if (r.ok) (await r.json()).forEach(u=>{ const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; uSel.appendChild(o); });

  } catch {}

  const listWrap=mk('div','panel-list'); body.appendChild(listWrap);

  filterBtn.addEventListener('click', ()=>renderFeedbackList(listWrap));

  await renderFeedbackList(listWrap);

}



async function renderFeedbackList(container) {

  container.innerHTML='<p class="admin-empty">Loading…</p>';

  const userId=document.getElementById('af-user-filter')?.value||'';

  const rating=document.getElementById('af-rating-filter')?.value||'';

  const qs=new URLSearchParams();

  if (userId) qs.set('user_id', userId);

  if (rating) qs.set('rating', rating);

  try {

    const r=await fetch(`/classes/${currentClassId}/feedback?${qs}`,{headers:getHeaders()});

    if (!r.ok) { container.innerHTML='<p class="admin-empty">Failed to load.</p>'; return; }

    const fb=await r.json();

    container.innerHTML='';

    if (!fb.length) { container.innerHTML='<p class="admin-empty">No feedback yet.</p>'; return; }

    fb.forEach(f=>{

      const el=mk('div','fb-entry');

      const ratingLbl=f.feedback_rating==='up'?'👍 Helpful':'👎 Not helpful';

      const topicLabel=f.topic_id?(topics.find(t=>t.id===f.topic_id)?.name||''):'' ;

      el.innerHTML=`

        <div class="fbe-head">

          <span class="fbe-user">${esc(f.username)}</span>

          <span class="fbe-rating ${f.feedback_rating}">${ratingLbl}</span>

          ${topicLabel?`<span class="fbe-topic">${esc(topicLabel)}</span>`:''}

          <span class="fbe-date">${new Date(f.created_at).toLocaleString()}</span>

        </div>

        ${f.feedback_comment?`<div class="fbe-comment">${esc(f.feedback_comment)}</div>`:''}`;

      container.appendChild(el);

    });

  } catch { container.innerHTML='<p class="admin-empty">Error loading feedback.</p>'; }

}



/* ─────────────────────── top-k ─────────────────────── */



let topK=parseInt(localStorage.getItem(LS_TOPK)||'5');

const topkVal=document.getElementById('topk-val');

topkVal.textContent=topK;

document.getElementById('topk-dec').addEventListener('click', ()=>{ if(topK>1){topK--;topkVal.textContent=topK;localStorage.setItem(LS_TOPK,topK);} });

document.getElementById('topk-inc').addEventListener('click', ()=>{ if(topK<20){topK++;topkVal.textContent=topK;localStorage.setItem(LS_TOPK,topK);} });



/* ─────────────────────── New chat ─────────────────────── */



document.getElementById('new-chat-btn').addEventListener('click', ()=>{

  currentChatId=null; currentTopicId=null;

  renderTopicChip(); renderSessions(); clearOutput();

});



function clearOutput() { document.getElementById('output').innerHTML=''; appendEmpty(); }



function appendEmpty() {

  const out=document.getElementById('output');

  const es=mk('div',''); es.id='empty-state';

  const role=getUser().role||'student';

  const bodyText=role==='student'

    ?'Pick a topic from the composer below, then ask your first question.'

    :'Open the Admin panel, create a topic, upload materials, then start a conversation.';

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



/* ─────────────────────── Query ─────────────────────── */



const output        =document.getElementById('output');

const questionInput =document.getElementById('question-input');

const askBtn        =document.getElementById('ask-btn');

const askLabel      =document.getElementById('ask-label');



questionInput.addEventListener('input', ()=>{

  questionInput.style.height='auto';

  questionInput.style.height=Math.min(questionInput.scrollHeight,180)+'px';

});

questionInput.addEventListener('keydown', e=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();submitQuery();} });

askBtn.addEventListener('click', submitQuery);



const STEPS=['hypothesis','sources','answering'];

const SLABELS={hypothesis:'Thinking',sources:'Searching',answering:'Writing'};



function makePipeline() {

  const wrap=mk('div','pipeline'), els={};

  STEPS.forEach((s,i)=>{

    const step=mk('span','pip-step'), dot=mk('span','pip-dot');

    step.append(dot, document.createTextNode(SLABELS[s]));

    wrap.appendChild(step); els[s]=step;

    if (i<STEPS.length-1) { const c=mk('span','pip-connector'); c.dataset.step=s; wrap.appendChild(c); }

  });

  return {wrap, els};

}

function setPipActive(els,wrap,step) {

  const idx=STEPS.indexOf(step);

  STEPS.forEach((s,i)=>{ els[s].className=i<idx?'pip-step done':i===idx?'pip-step active':'pip-step'; });

  wrap.querySelectorAll('.pip-connector').forEach((c,i)=>c.classList.toggle('done',i<idx));

}

function setPipDone(els,wrap) {

  STEPS.forEach(s=>{ els[s].className='pip-step done'; });

  wrap.querySelectorAll('.pip-connector').forEach(c=>c.classList.add('done'));

}



function getInner() {

  let inner=output.querySelector('.output-inner');

  if (!inner) { inner=mk('div','output-inner'); output.appendChild(inner); }

  return inner;

}



async function submitQuery() {

  const question=questionInput.value.trim();

  if (!question||askBtn.disabled) return;

  if (!currentClassId) return;



  // create chat on server if this is a new conversation

  if (!currentChatId) {

    try {

      const cr=await fetch(`/classes/${currentClassId}/chats`,{

        method:'POST', headers:getHeaders({'Content-Type':'application/json'}),

        body:JSON.stringify({topic_id:currentTopicId||null}),

      });

      if (!cr.ok) return;

      currentChatId=(await cr.json()).id;

    } catch { return; }

  }



  const es=document.getElementById('empty-state');

  if (es) es.remove();

  setLocked(true);

  questionInput.value=''; questionInput.style.height='auto';



  const inner=getInner();

  const block=mk('div','response');

  const qb=mk('div','q-bubble'); qb.textContent=question;

  const {wrap:pipWrap, els:pipEls}=makePipeline();

  setPipActive(pipEls,pipWrap,'hypothesis');

  const srcSec=mk('div'); srcSec.style.display='none';

  const srcGrid=mk('div','source-grid');

  srcSec.append(lbl('Sources used'), srcGrid);

  const thinkWrap=mk('div','thinking-wrap');

  [72,58,40].forEach((w,i)=>{ const l=mk('div','shimmer-line'); l.style.cssText=`width:${w}%;animation-delay:${i*0.18}s`; thinkWrap.appendChild(l); });

  const ansSec=mk('div'); ansSec.style.display='none';

  const ansText=mk('div','answer-text');

  const cursor=mk('span','cursor');

  ansSec.appendChild(ansText);

  block.append(qb, pipWrap, srcSec, thinkWrap, ansSec);

  inner.appendChild(block);

  scrollEnd();



  let collAnswer='', collSources=[], collHyde='', asstMsgId=null;



  try {

    const res=await fetch(`/chats/${currentChatId}/query/stream`,{

      method:'POST',

      headers:getHeaders({'Content-Type':'application/json'}),

      body:JSON.stringify({question, top_k:topK}),

    });

    if (!res.ok) {

      thinkWrap.remove(); ansSec.style.display='block';

      ansText.textContent=await friendlyError(res);

      setPipDone(pipEls,pipWrap); return;

    }

    const reader=res.body.getReader(), decoder=new TextDecoder();

    let buf='';

    while (true) {

      const {done,value}=await reader.read();

      if (done) break;

      buf+=decoder.decode(value,{stream:true});

      const parts=buf.split('\n\n'); buf=parts.pop();

      for (const part of parts) {

        if (!part.startsWith('data: ')) continue;

        let evt; try { evt=JSON.parse(part.slice(6)); } catch { continue; }

        if (evt.type==='hyde') {

          collHyde=evt.text; setPipActive(pipEls,pipWrap,'sources');

        } else if (evt.type==='sources') {

          collSources=evt.sources;

          thinkWrap.remove(); srcSec.style.display='block';

          ansSec.style.display='block'; ansText.appendChild(cursor);

          setPipActive(pipEls,pipWrap,'answering');

          evt.sources.forEach((s,i)=>{

            const pct=Math.round(s.relevance*100);

            const card=mk('div','source-card'); card.style.animationDelay=(i*0.06)+'s';

            card.innerHTML=`<div class="sc-top"><span class="sc-rank">Source ${i+1}</span><span class="sc-pct">${pct}%</span></div>

              <div class="sc-name" title="${esc(s.source)}">${esc(s.source)}</div>

              <div class="sc-page">Page ${s.page}</div>

              <div class="rel-track"><div class="rel-fill"></div></div>`;

            srcGrid.appendChild(card);

            requestAnimationFrame(()=>{ card.querySelector('.rel-fill').style.width=pct+'%'; });

          });

          scrollEnd();

        } else if (evt.type==='delta') {

          collAnswer+=evt.text;

          ansText.insertBefore(document.createTextNode(evt.text),cursor);

          scrollEnd();

        } else if (evt.type==='error') {

          cursor.remove();

          ansText.appendChild(document.createTextNode(`\n\n⚠ ${evt.message}`));

          setPipDone(pipEls,pipWrap);

        } else if (evt.type==='done') {

          cursor.remove(); setPipDone(pipEls,pipWrap);

          asstMsgId=evt.message_id;

          if (asstMsgId) appendFeedbackBar(block, currentChatId, asstMsgId, null);

          await fetchSessions();

        }

      }

    }

  } catch {

    thinkWrap.remove(); cursor.remove(); ansSec.style.display='block';

    ansText.appendChild(document.createTextNode(ansText.textContent?'\n\n⚠ Connection lost.':'Network error — is the server running?'));

    setPipDone(pipEls,pipWrap);

  } finally {

    setLocked(false);

    questionInput.focus();

  }

}



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



function scrollEnd() { requestAnimationFrame(()=>{ output.scrollTop=output.scrollHeight; }); }

function mk(tag,cls)  { const e=document.createElement(tag); if(cls) e.className=cls; return e; }

function lbl(text)    { const e=mk('p','blk-label'); e.textContent=text; return e; }

function esc(s)       { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function trunc(s,n)   { return s.length>n?s.slice(0,n)+'…':s; }



async function friendlyError(res) {

  let d=''; try { d=(await res.json()).detail||''; } catch {}

  if (res.status===401) { setTimeout(showLoginModal,150); return '🔒 Session expired — please sign in again.'; }

  if (res.status===403) return '🚫 Access denied.';

  if (res.status===400) return d||'Invalid request.';

  if (res.status===500) return `Server error: ${d||'something went wrong.'}`;

  return `Unexpected error (${res.status})${d?': '+d:''}.`;

}



/* ─────────────────────── Feedback bar ─────────────────────── */



function appendFeedbackBar(block, chatId, msgId, existing) {

  const bar=mk('div','feedback-bar');

  const upBtn=mk('button','fb-btn up');   upBtn.innerHTML='👍'; upBtn.title='Helpful';

  const downBtn=mk('button','fb-btn down'); downBtn.innerHTML='👎'; downBtn.title='Not helpful';

  const statusEl=mk('span','fb-status');

  const commentRow=mk('div','fb-comment-row');

  const commentIn=mk('input','fb-comment-input');

  commentIn.placeholder='Optional comment…'; commentIn.type='text';

  const submitBtn=mk('button','fb-submit'); submitBtn.textContent='Submit';

  commentRow.append(commentIn, submitBtn);

  let pendingRating=null;

  if (existing) {

    if (existing.rating==='up')   upBtn.classList.add('active');

    if (existing.rating==='down') downBtn.classList.add('active');

    if (existing.comment) statusEl.textContent=`"${trunc(existing.comment,60)}"`;

  }

  function selectRating(r) { pendingRating=r; upBtn.classList.toggle('active',r==='up'); downBtn.classList.toggle('active',r==='down'); commentRow.classList.add('visible'); commentIn.focus(); }

  upBtn.addEventListener('click', ()=>selectRating('up'));

  downBtn.addEventListener('click', ()=>selectRating('down'));

  submitBtn.addEventListener('click', async ()=>{

    if (!pendingRating) return;

    const comment=commentIn.value.trim();

    submitBtn.disabled=true; statusEl.textContent='Saving…';

    try {

      const r=await fetch(`/chats/${chatId}/messages/${msgId}/feedback`,{

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



/* ─────────────────────── Admin panels ─────────────────────── */



function showAdminPanel(name) {

  document.getElementById('panel-title').textContent={topics:'Topics',classes:'Classes',chats:'Browse Chats',feedback:'Feedback',users:'Users'}[name]||name;

  document.getElementById('chat-view').style.display='none';

  document.getElementById('admin-panel').style.display='flex';

  if (name==='topics')        loadTopicsPanel();

  else if (name==='classes')  loadClassesPanel();

  else if (name==='chats')    loadChatsPanel();

  else if (name==='feedback') loadFeedbackPanel();

}

function hideAdminPanel() {

  document.getElementById('admin-panel').style.display='none';

  document.getElementById('chat-view').style.display='flex';

}



document.getElementById('admin-toggle').addEventListener('click', ()=>document.getElementById('sb-admin').classList.toggle('open'));

document.getElementById('manage-topics-btn').addEventListener('click', ()=>showAdminPanel('topics'));

document.getElementById('manage-classes-btn').addEventListener('click', ()=>showAdminPanel('classes'));

document.getElementById('manage-users-btn').addEventListener('click', showUsersModal);

document.getElementById('browse-chats-btn').addEventListener('click', ()=>showAdminPanel('chats'));

document.getElementById('view-feedback-btn').addEventListener('click', ()=>showAdminPanel('feedback'));

document.getElementById('panel-back-btn').addEventListener('click', hideAdminPanel);

document.querySelectorAll('.reveal-btn').forEach(btn=>{
  const EYE='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const EYE_OFF='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
  btn.innerHTML=EYE;
  btn.addEventListener('click', ()=>{
    const inp=document.getElementById(btn.dataset.for);
    const show=inp.type==='password'; inp.type=show?'text':'password'; btn.innerHTML=show?EYE_OFF:EYE;
  });
});



/* ─────────────────────── Init ─────────────────────── */



async function init() {

  setConnState('checking');

  try {

    const r=await fetch('/health',{signal:AbortSignal.timeout(4000)});

    if (!r.ok) { showOffline(); return; }

    setConnState('online');

    const ar=await fetch('/auth/me',{headers:getHeaders()});

    if (!ar.ok) { showLoginModal(); return; }

    const user=await ar.json();

    storeUser({id:user.id, username:user.username, role:user.role});

    updateUserBar(user.username, user.role);

    myClasses=await fetchClasses();

    const savedId=localStorage.getItem(LS_CLASS);

    const saved=myClasses.find(c=>c.id===savedId);

    if (saved) await selectClass(saved.id);

    else if (myClasses.length===1) await selectClass(myClasses[0].id);

    else if (myClasses.length>1)  showClassPicker(true);

  } catch { showOffline(); }

}



/* ── Boot ── */

(function() {

  const mac=/Mac|iPhone|iPad|iPod/.test(navigator.platform);

  const hint=document.getElementById('send-hint');

  if (hint) hint.innerHTML=mac?'<kbd>⌘</kbd><kbd>↵</kbd> to send':'<kbd>Ctrl</kbd><kbd>↵</kbd> to send';

})();

init();



