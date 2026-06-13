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

/* ── Storage keys ── */

const LS_TOKEN = 'rag_auth_token';
const LS_USER  = 'rag_user_info';
const LS_CLASS = 'rag_class_id';

/* ── Storage helpers ── */

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

/* ── DOM helpers ── */

function mk(tag, cls)  { const e=document.createElement(tag); if(cls) e.className=cls; return e; }
function lbl(text)     { const e=mk('p','blk-label'); e.textContent=text; return e; }
function esc(s)        { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function trunc(s, n)   { return s.length>n ? s.slice(0,n)+'…' : s; }

function relTime(iso) {
  const s=Math.floor((Date.now()-new Date(iso))/1000);
  if (s<60) return 'just now';
  const m=Math.floor(s/60); if (m<60) return m+'m ago';
  const h=Math.floor(m/60); if (h<24) return h+'h ago';
  const d=Math.floor(h/24); if (d<7)  return d+'d ago';
  return new Date(iso).toLocaleDateString();
}

/* ── friendlyError ── */

async function friendlyError(res) {
  let d=''; try { d=(await res.json()).detail||''; } catch {}
  if (res.status===401) { setTimeout(showLoginModal,150); return '🔒 Session expired — please sign in again.'; }
  if (res.status===403) return '🚫 Access denied.';
  if (res.status===400) return d||'Invalid request.';
  if (res.status===500) return `Server error: ${d||'something went wrong.'}`;
  return `Unexpected error (${res.status})${d?': '+d:''}.`;
}
