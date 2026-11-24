param([string]$Root = '')
$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Join-Path $PSScriptRoot '../receipt_labeled' }
$Root = (Resolve-Path -LiteralPath $Root).Path
$labels = @(Get-ChildItem -LiteralPath (Join-Path $Root 'labels') -Filter 'r*.json' | Sort-Object Name | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json })
$json = ConvertTo-Json -InputObject $labels -Depth 40 -Compress
$json = $json.Replace('<','\u003c').Replace('>','\u003e').Replace('&','\u0026')
$html = @'
<!doctype html><html lang="ko"><meta charset="utf-8"><title>영수증 라벨 검수</title>
<style>body{font:16px system-ui;margin:0;color:#182431;background:#f4f6f8}header{padding:16px;background:white;position:sticky;top:0;z-index:2;border-bottom:1px solid #ccd4dd}button,input,select{font:inherit;padding:8px;margin:3px}button{cursor:pointer}main{display:grid;grid-template-columns:42% 58%}aside{position:sticky;top:140px;height:80vh;overflow:auto}aside img{width:100%}#lines{padding:12px}.line{padding:12px;background:white;margin-bottom:10px;border:2px solid transparent}.line.ok{border-color:#23834a}.line img{max-width:100%;max-height:140px;display:block;background:#eee}.line textarea{width:96%;font:18px system-ui;min-height:40px;margin-top:8px}.note{font-size:13px;color:#45566a}#warning{color:#a43b17}a{color:#175db2}@media(max-width:800px){main{display:block}aside{position:static;height:35vh}}</style>
<header><b>영수증 라벨 검수</b> <span id="progress"></span><br>
<button id="prev">← 이전</button><select id="receipt"></select><button id="next">다음 →</button>
<input id="search" placeholder="r001 / 원본 파일명 검색"><button id="find">찾기</button>
<button id="export">수정본 저장(JSON)</button><label>수정본 불러오기<input id="import" type="file" accept=".json"></label>
<div class="note" id="warning">자동 라벨은 검수 전입니다. 각 줄의 이미지와 글자를 대조한 뒤 ‘확인’을 체크하세요. 판독 불가·잘린 줄은 ‘제외’를 선택하세요. 이 화면은 누락된 영역을 추가하지 않으므로 전체 검수 완료를 의미하지 않습니다.</div>
<div class="note">변경은 이 브라우저에 임시 보관됩니다. ‘수정본 저장(JSON)’ 후 apply_receipt_review.ps1로 반영해야 파일과 학습 목록에 적용됩니다.</div></header>
<main><aside><a id="original" target="_blank">큰 이미지 열기</a><img id="image"></aside><section id="lines"></section></main>
<script>
const receipts=__DATA__;
const storageKey='receipt_labels_review_v1';let edits={};try{edits=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
let current=0;const $=s=>document.querySelector(s);const select=$('#receipt');
for(const r of receipts){let o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · '+r.original_name;select.append(o)}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(edits));$('#warning').textContent='임시 저장됨. 파일 반영을 위해 수정본 JSON을 저장하세요.'}catch(e){$('#warning').textContent='브라우저 임시 저장 실패. 지금 수정본 JSON을 저장하세요.'}progress()}
function progress(){let checked=receipts.reduce((n,r)=>n+r.lines.filter(l=>(edits[l.id]?.status||l.status)==='verified').length,0);$('#progress').textContent=receipts.length+'장 · 확인 '+checked+'줄'}
function render(){const r=receipts[current];select.value=r.id;$('#image').src=r.image;$('#original').href=r.image;$('#lines').replaceChildren();
for(const l of r.lines){let saved=edits[l.id]||{text:l.text,status:l.status,receipt_id:r.id,source_sha256:r.source_sha256};let div=document.createElement('div');div.className='line'+(saved.status==='verified'?' ok':'');let title=document.createElement('b');title.textContent=l.id;let img=document.createElement('img');img.src=l.crop;img.loading='lazy';let t=document.createElement('textarea');t.value=saved.text;let state=document.createElement('select');for(const [value,label]of [['needs_review','검수 전'],['verified','확인'],['excluded','제외(판독 불가 등)']]){let o=document.createElement('option');o.value=value;o.textContent=label;state.append(o)}state.value=saved.status;
function save(){edits[l.id]={text:t.value,status:state.value,receipt_id:r.id,source_sha256:r.source_sha256};div.classList.toggle('ok',state.value==='verified');persist()}
t.addEventListener('input',()=>{state.value='needs_review';save()});state.addEventListener('change',save);div.append(title,img,t,state);$('#lines').append(div)}progress();window.scrollTo(0,0)}
select.onchange=()=>{current=receipts.findIndex(r=>r.id===select.value);render()};$('#prev').onclick=()=>{current=Math.max(0,current-1);render()};$('#next').onclick=()=>{current=Math.min(receipts.length-1,current+1);render()};$('#find').onclick=()=>{let q=$('#search').value.toLowerCase();let n=receipts.findIndex(r=>r.id===q||r.original_name.toLowerCase().includes(q));if(n>=0){current=n;render()}};
$('#export').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema_version:1,edits},null,2)],{type:'application/json'}));a.download='receipt_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};
$('#import').onchange=async e=>{try{let p=JSON.parse(await e.target.files[0].text());if(p.schema_version!==1||!p.edits)throw Error('수정본 형식 오류');const valid=new Map(receipts.flatMap(r=>r.lines.map(l=>[l.id,r])));for(const[id,v]of Object.entries(p.edits)){const r=valid.get(id);if(!r||v.receipt_id!==r.id||v.source_sha256!==r.source_sha256||typeof v.text!=='string'||!['needs_review','verified','excluded'].includes(v.status))throw Error('이미지 또는 라벨 불일치: '+id)}edits={...edits,...p.edits};persist();render()}catch(err){alert(err.message)}};
render();
</script></html>
'@
$html = $html.Replace('__DATA__', $json)
[IO.File]::WriteAllText((Join-Path $Root 'review.html'),$html,(New-Object Text.UTF8Encoding($false)))
Write-Output "Review page: $Root/review.html ($($labels.Count) receipts)"
