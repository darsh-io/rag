'use strict';

/* ── Users modal ── */

const usersModal = document.getElementById('users-modal');
function showUsersModal() { loadUserList(); usersModal.classList.add('open'); }
function hideUsersModal() { usersModal.classList.remove('open'); }

document.getElementById('close-users-btn').addEventListener('click', hideUsersModal);

async function loadUserList() {
  const list = document.getElementById('user-list');
  list.innerHTML = '<p style="font-size:12px;color:var(--text-muted);padding:6px 4px">Loading…</p>';
  try {
    const r = await fetch('/users', {headers:getHeaders()});
    if (!r.ok) { list.innerHTML='<p style="font-size:12px;color:var(--red);padding:6px 4px">Failed to load users.</p>'; return; }
    const data = await r.json();
    const users = data.items ?? data;
    const me = getUser().id;
    list.innerHTML = '';
    users.forEach(u => {
      const el = mk('div','user-row');
      el.innerHTML = `
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
      if (u.id !== me) {
        el.querySelector('.ur-toggle').addEventListener('click', async () => {
          const path = u.is_active ? 'deactivate' : 'activate';
          await fetch(`/users/${u.id}/${path}`, {method:'PATCH', headers:getHeaders()});
          loadUserList();
        });
        el.querySelector('.ur-del').addEventListener('click', async () => {
          if (!await showConfirm(`Permanently delete "${u.username}"?\n\nThis cannot be undone. Their chats and enrollment history will be erased.`, 'Delete')) return;
          await fetch(`/users/${u.id}`, {method:'DELETE', headers:getHeaders()});
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
  const username = document.getElementById('new-user-username').value.trim();
  const password = document.getElementById('new-user-password').value;
  const role     = document.getElementById('new-user-role').value;
  const err      = document.getElementById('add-user-error');
  err.style.display = 'none';
  if (!username) { err.textContent='Username required.'; err.style.display='block'; return; }
  if (!password) { err.textContent='Password required.'; err.style.display='block'; return; }
  try {
    const r = await fetch('/users', {
      method:'POST', headers:getHeaders({'Content-Type':'application/json'}),
      body:JSON.stringify({username, password, role}),
    });
    if (!r.ok) { const d=await r.json(); err.textContent=d.detail||'Failed to create user.'; err.style.display='block'; return; }
    document.getElementById('new-user-username').value = '';
    document.getElementById('new-user-password').value = '';
    loadUserList();
  } catch { err.textContent='Cannot reach server.'; err.style.display='block'; }
}

/* ── Topics panel ── */

async function loadTopicsPanel() {
  const body = document.getElementById('panel-body');
  body.innerHTML = '';
  const createRow = mk('div','tp-create-row');
  const nameInput = document.createElement('input');
  nameInput.type='text'; nameInput.placeholder='New topic name…'; nameInput.autocomplete='off'; nameInput.spellcheck=false;
  const createBtn = document.createElement('button'); createBtn.textContent='Create topic';
  const cardsWrap = mk('div','tp-cards');
  createBtn.addEventListener('click', async () => {
    const name = nameInput.value.trim(); if (!name) return;
    const r = await fetch(`/classes/${currentClassId}/topics`, {
      method:'POST', headers:getHeaders({'Content-Type':'application/json'}),
      body:JSON.stringify({name}),
    });
    if (!r.ok) return;
    nameInput.value = '';
    await fetchTopics(); renderTopicCards(cardsWrap);
  });
  nameInput.addEventListener('keydown', e=>{ if(e.key==='Enter') createBtn.click(); });
  createRow.append(nameInput, createBtn);
  body.append(createRow, cardsWrap);
  await fetchTopics();
  renderTopicCards(cardsWrap);
}

function renderTopicCards(container) {
  container.innerHTML = '';
  if (!topics.length) {
    const empty = mk('p','tp-empty-files'); empty.textContent='No topics yet. Create one above.';
    container.appendChild(empty); return;
  }
  topics.forEach(topic => {
    const card   = mk('div','tp-card'); card.dataset.topicId=topic.id;
    const hdr    = mk('div','tp-card-hdr');
    const nameEl = mk('span','tp-card-name'); nameEl.textContent=topic.name;
    const delBtn = mk('button','tp-card-del'); delBtn.title='Delete topic';
    delBtn.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/></svg>`;
    delBtn.addEventListener('click', async () => {
      if (!await showConfirm(`Delete topic "${topic.name}"?`, 'Delete')) return;
      await fetch(`/classes/${currentClassId}/topics/${topic.id}`, {method:'DELETE', headers:getHeaders()});
      await fetchTopics(); renderTopicCards(document.querySelector('.tp-cards'));
    });
    hdr.append(nameEl, delBtn); card.appendChild(hdr);
    const filesLbl  = mk('p','tp-files-lbl'); filesLbl.textContent='Files'; card.appendChild(filesLbl);
    const filesWrap = mk('div','tp-files'); filesWrap.dataset.topicId=topic.id;
    card.appendChild(filesWrap);
    loadDocumentTags(filesWrap, topic.id);
    const drop       = mk('div','tp-drop');
    const fileInput  = document.createElement('input'); fileInput.type='file'; fileInput.multiple=true;
    fileInput.accept='.pdf,.txt,.md,.docx,.html,.htm,.epub,.odt';
    const dropLabel  = mk('span','tp-drop-label'); dropLabel.textContent='Click to upload files — or drag them here';
    drop.append(fileInput, dropLabel);
    const uploadingWrap = mk('div','tp-uploading');
    drop.addEventListener('click',    e=>{ if(e.target!==fileInput) fileInput.click(); });
    drop.addEventListener('dragover', e=>{ e.preventDefault(); drop.classList.add('drag-over'); });
    drop.addEventListener('dragleave', ()=>drop.classList.remove('drag-over'));
    drop.addEventListener('drop', e=>{ e.preventDefault(); drop.classList.remove('drag-over'); handleFiles(Array.from(e.dataTransfer.files), topic.id, uploadingWrap, filesWrap); });
    fileInput.addEventListener('change', ()=>{ if (!fileInput.files.length) return; handleFiles(Array.from(fileInput.files), topic.id, uploadingWrap, filesWrap); fileInput.value=''; });
    card.appendChild(drop);
    card.appendChild(uploadingWrap);
    container.appendChild(card);
  });
}

async function loadDocumentTags(filesWrap, topicId) {
  filesWrap.innerHTML = '';
  try {
    const r = await fetch(`/classes/${currentClassId}/topics/${topicId}/documents`, {headers:getHeaders()});
    if (!r.ok) { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'No files yet.'})); return; }
    const docs = await r.json();
    if (!docs.length) { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'No files yet.'})); return; }
    docs.forEach(doc => {
      const tag = mk('span','tp-src-tag');
      tag.innerHTML=`${esc(doc.filename)}<button class="tp-src-del" title="Delete">×</button>`;
      tag.querySelector('.tp-src-del').addEventListener('click', async e => {
        e.stopPropagation();
        if (!await showConfirm(`Delete "${doc.filename}"?`, 'Delete')) return;
        await fetch(`/classes/${currentClassId}/topics/${topicId}/documents/${doc.id}`, {method:'DELETE', headers:getHeaders()});
        loadDocumentTags(filesWrap, topicId);
      });
      filesWrap.appendChild(tag);
    });
  } catch { filesWrap.appendChild(Object.assign(mk('span','tp-empty-files'),{textContent:'Error loading files.'})); }
}

function handleFiles(files, topicId, uploadingWrap, filesWrap) {
  files.forEach(file => {
    const pill = mk('div','tp-upload-pill'); pill.textContent=`${file.name} — Uploading…`;
    uploadingWrap.appendChild(pill);
    ingestAndAssign(file, topicId, pill, filesWrap);
  });
}

async function ingestAndAssign(file, topicId, pill, filesWrap) {
  const form = new FormData(); form.append('file', file);
  try {
    const r = await fetch(`/classes/${currentClassId}/topics/${topicId}/documents`, {method:'POST', headers:getHeaders(), body:form});
    if (!r.ok) { pill.textContent=`${file.name} — Upload failed.`; pill.classList.add('error'); return; }
    const { id: docId } = await r.json();
    pill.textContent = `${file.name} — Processing…`;

    // Poll until status is 'ready' or 'error'
    while (true) {
      await new Promise(res => setTimeout(res, 3000));
      try {
        const sr = await fetch(`/classes/${currentClassId}/topics/${topicId}/documents/${docId}/status`, {headers:getHeaders()});
        if (!sr.ok) break;
        const s = await sr.json();
        if (s.status === 'ready') break;
        if (s.status === 'error') {
          pill.textContent = `${file.name} — Error: ${s.error_message||'ingestion failed'}`;
          pill.classList.add('error');
          return;
        }
      } catch { break; }
    }

    pill.textContent=`✓ ${file.name}`; pill.classList.add('done');
    await fetchTopics();
    loadDocumentTags(filesWrap, topicId);
    setTimeout(()=>{ pill.style.transition='opacity 0.5s'; pill.style.opacity='0'; setTimeout(()=>pill.remove(),500); }, 3000);
  } catch { pill.textContent=`${file.name} — Upload failed.`; pill.classList.add('error'); }
}

/* ── Classes panel ── */

async function loadClassesPanel() {
  const body = document.getElementById('panel-body');
  body.innerHTML = '';
  const createRow = mk('div','tp-create-row');
  const nameInput = document.createElement('input');
  nameInput.type='text'; nameInput.placeholder='New class name…'; nameInput.autocomplete='off';
  const createBtn = document.createElement('button'); createBtn.textContent='Create class';
  const cardsWrap = mk('div','tp-cards');
  createBtn.addEventListener('click', async () => {
    const name = nameInput.value.trim(); if (!name) return;
    const r = await fetch('/classes', {method:'POST', headers:getHeaders({'Content-Type':'application/json'}), body:JSON.stringify({name})});
    if (!r.ok) return;
    nameInput.value = '';
    myClasses = await fetchClasses();
    renderClassCards(cardsWrap);
  });
  nameInput.addEventListener('keydown', e=>{ if(e.key==='Enter') createBtn.click(); });
  createRow.append(nameInput, createBtn);
  body.append(createRow, cardsWrap);
  await renderClassCards(cardsWrap);
}

async function renderClassCards(container) {
  container.innerHTML='<p style="font-size:12px;color:var(--text-muted);padding:6px">Loading…</p>';
  try {
    const [clsRes, usrRes] = await Promise.all([
      fetch('/classes', {headers:getHeaders()}),
      fetch('/users',   {headers:getHeaders()}),
    ]);
    const allClasses = clsRes.ok ? await clsRes.json() : [];
    const usrData    = usrRes.ok ? await usrRes.json() : [];
    const allUsers   = usrData.items ?? usrData;
    const teachers   = allUsers.filter(u=>u.role==='teacher'&&u.is_active);
    const students   = allUsers.filter(u=>u.role==='student'&&u.is_active);
    container.innerHTML = '';
    if (!allClasses.length) { container.innerHTML='<p style="font-size:12px;color:var(--text-hint);font-style:italic">No classes yet.</p>'; return; }
    for (const cls of allClasses) {
      const card   = mk('div','cls-card');
      const hdr    = mk('div','cls-card-hdr');
      const nameEl = mk('span','cls-card-name'); nameEl.textContent=cls.name;
      if (!cls.is_active) {
        card.style.opacity='0.5';
        const badge = mk('span','cls-inactive-badge'); badge.textContent='Inactive'; hdr.appendChild(badge);
      }
      const toggleBtn = mk('button', cls.is_active?'cls-card-del':'cls-card-act');
      toggleBtn.title = cls.is_active ? 'Deactivate class' : 'Activate class';
      toggleBtn.innerHTML = cls.is_active
        ?`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`
        :`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`;
      toggleBtn.addEventListener('click', async () => {
        const action = cls.is_active ? 'deactivate' : 'activate';
        if (cls.is_active && !await showConfirm(`Deactivate class "${cls.name}"?\n\nStudents will lose access until it is re-activated.`, 'Deactivate')) return;
        await fetch(`/classes/${cls.id}/${action}`, {method:'PATCH', headers:getHeaders()});
        myClasses = await fetchClasses(); renderClassCards(container);
      });
      hdr.append(nameEl, toggleBtn); card.appendChild(hdr);
      const teachersWrap = mk('div'); card.appendChild(teachersWrap);
      await renderMembersSection(teachersWrap, cls.id, 'teacher', 'Teachers', teachers);
      const studentsWrap = mk('div'); card.appendChild(studentsWrap);
      await renderMembersSection(studentsWrap, cls.id, 'student', 'Students', students);
      container.appendChild(card);
    }
  } catch(e) { container.innerHTML=`<p style="font-size:12px;color:var(--red)">${esc(String(e))}</p>`; }
}

async function renderMembersSection(container, classId, memberType, label, candidates) {
  container.innerHTML = '';
  const l = mk('p','cls-section-lbl'); l.textContent=label; container.appendChild(l);
  const membersDiv = mk('div','cls-members'); container.appendChild(membersDiv);
  const endpoint   = memberType==='teacher' ? 'teachers' : 'students';
  try {
    const r       = await fetch(`/classes/${classId}/${endpoint}`, {headers:getHeaders()});
    const members = r.ok ? await r.json() : [];
    if (members.length) {
      members.forEach(m => {
        const tag = mk('span','cls-member-tag');
        tag.innerHTML=`${esc(m.username)}<button class="cls-member-del" title="Remove">×</button>`;
        tag.querySelector('.cls-member-del').addEventListener('click', async () => {
          await fetch(`/classes/${classId}/${endpoint}/${m.id}`, {method:'DELETE', headers:getHeaders()});
          renderMembersSection(container, classId, memberType, label, candidates);
        });
        membersDiv.appendChild(tag);
      });
    } else {
      membersDiv.appendChild(Object.assign(mk('span','cls-empty'), {textContent:`No ${label.toLowerCase()} yet.`}));
    }
    const addRow = mk('div','cls-add-row');
    const sel    = document.createElement('select'); sel.innerHTML=`<option value="">Add ${memberType}…</option>`;
    candidates.forEach(u => { const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; sel.appendChild(o); });
    const addBtn = mk('button','cls-add-btn'); addBtn.textContent='Add';
    addBtn.addEventListener('click', async () => {
      if (!sel.value) return;
      await fetch(`/classes/${classId}/${endpoint}`, {method:'POST', headers:getHeaders({'Content-Type':'application/json'}), body:JSON.stringify({user_id:sel.value})});
      renderMembersSection(container, classId, memberType, label, candidates);
    });
    addRow.append(sel, addBtn); container.appendChild(addRow);
  } catch {}
}

/* ── Chats panel ── */

async function loadChatsPanel() {
  const body = document.getElementById('panel-body');
  body.innerHTML = '';
  const filterRow = mk('div','panel-filter-row');
  const uSel = document.createElement('select'); uSel.id='ac-user-filter';
  uSel.innerHTML = '<option value="">All users</option>';
  const tSel = document.createElement('select'); tSel.id='ac-topic-filter';
  tSel.innerHTML = '<option value="">All topics</option>';
  topics.forEach(t => { const o=document.createElement('option'); o.value=t.id; o.textContent=t.name; tSel.appendChild(o); });
  const filterBtn = document.createElement('button'); filterBtn.textContent='Filter';
  filterRow.append(uSel, tSel, filterBtn);
  body.appendChild(filterRow);
  try {
    const r = await fetch(`/classes/${currentClassId}/students`, {headers:getHeaders()});
    if (r.ok) (await r.json()).forEach(u => { const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; uSel.appendChild(o); });
  } catch {}
  const listWrap = mk('div','panel-list'); body.appendChild(listWrap);
  filterBtn.addEventListener('click', ()=>renderChatsList(listWrap));
  await renderChatsList(listWrap);
}

async function renderChatsList(container) {
  container.innerHTML = '<p class="admin-empty">Loading…</p>';
  const userId  = document.getElementById('ac-user-filter')?.value||'';
  const topicId = document.getElementById('ac-topic-filter')?.value||'';
  const qs = new URLSearchParams();
  if (userId)  qs.set('user_id', userId);
  if (topicId) qs.set('topic_id', topicId);
  try {
    const r = await fetch(`/classes/${currentClassId}/chats?${qs}`, {headers:getHeaders()});
    if (!r.ok) { container.innerHTML='<p class="admin-empty">Failed to load.</p>'; return; }
    const data = await r.json();
    const chats = data.items ?? data;
    container.innerHTML = '';
    if (!chats.length) { container.innerHTML='<p class="admin-empty">No chats found.</p>'; return; }
    chats.forEach(c => {
      const card       = mk('div','admin-chat-card');
      const topicLabel = c.topic_id ? (topics.find(t=>t.id===c.topic_id)?.name||c.topic_id) : 'All material';
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
  const body    = document.getElementById('panel-body');
  body.innerHTML = '';
  const backBtn = mk('button','admin-back-btn'); backBtn.innerHTML='← Back to list';
  backBtn.addEventListener('click', loadChatsPanel);
  const heading = mk('p','blk-label'); heading.textContent=`${username}: ${title||'Untitled'}`;
  body.append(backBtn, heading);
  try {
    const r = await fetch(`/classes/${classId}/chats/${chatId}/view`, {headers:getHeaders()});
    if (!r.ok) { body.appendChild(Object.assign(mk('p','admin-empty'),{textContent:'Failed to load chat.'})); return; }
    const chat = await r.json();
    const view = mk('div','admin-chat-view');
    const msgs = chat.messages || [];
    for (let i=0; i<msgs.length; i+=2) {
      const um=msgs[i]; const am=msgs[i+1]; if (!um) continue;
      const pair = mk('div','acv-pair');
      const qEl  = mk('div','acv-q'); qEl.textContent=um.content; pair.appendChild(qEl);
      if (am) {
        const aEl = mk('div','acv-a'); aEl.textContent=am.content; pair.appendChild(aEl);
        if (am.feedback_rating) {
          const fb = mk('span',`acv-fb ${am.feedback_rating}`);
          fb.textContent = am.feedback_rating==='up' ? '👍 Helpful' : '👎 Not helpful';
          if (am.feedback_comment) fb.textContent += ` — "${am.feedback_comment}"`;
          pair.appendChild(fb);
        }
      }
      view.appendChild(pair);
    }
    body.appendChild(view);
  } catch { body.appendChild(Object.assign(mk('p','admin-empty'),{textContent:'Error loading chat.'})); }
}

/* ── Feedback panel ── */

async function loadFeedbackPanel() {
  const body = document.getElementById('panel-body');
  body.innerHTML = '';
  const filterRow = mk('div','panel-filter-row');
  const uSel = document.createElement('select'); uSel.id='af-user-filter';
  uSel.innerHTML = '<option value="">All users</option>';
  const rSel = document.createElement('select'); rSel.id='af-rating-filter';
  rSel.innerHTML = '<option value="">All ratings</option><option value="up">👍 Positive</option><option value="down">👎 Negative</option>';
  const filterBtn = document.createElement('button'); filterBtn.textContent='Filter';
  filterRow.append(uSel, rSel, filterBtn);
  body.appendChild(filterRow);
  try {
    const r = await fetch(`/classes/${currentClassId}/students`, {headers:getHeaders()});
    if (r.ok) (await r.json()).forEach(u => { const o=document.createElement('option'); o.value=u.id; o.textContent=u.username; uSel.appendChild(o); });
  } catch {}
  const listWrap = mk('div','panel-list'); body.appendChild(listWrap);
  filterBtn.addEventListener('click', ()=>renderFeedbackList(listWrap));
  await renderFeedbackList(listWrap);
}

async function renderFeedbackList(container) {
  container.innerHTML = '<p class="admin-empty">Loading…</p>';
  const userId = document.getElementById('af-user-filter')?.value||'';
  const rating = document.getElementById('af-rating-filter')?.value||'';
  const qs = new URLSearchParams();
  if (userId) qs.set('user_id', userId);
  if (rating) qs.set('rating', rating);
  try {
    const r = await fetch(`/classes/${currentClassId}/feedback?${qs}`, {headers:getHeaders()});
    if (!r.ok) { container.innerHTML='<p class="admin-empty">Failed to load.</p>'; return; }
    const data = await r.json();
    const fb = data.items ?? data;
    container.innerHTML = '';
    if (!fb.length) { container.innerHTML='<p class="admin-empty">No feedback yet.</p>'; return; }
    fb.forEach(f => {
      const el         = mk('div','fb-entry');
      const ratingLbl  = f.feedback_rating==='up' ? '👍 Helpful' : '👎 Not helpful';
      const topicLabel = f.topic_id ? (topics.find(t=>t.id===f.topic_id)?.name||'') : '';
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
