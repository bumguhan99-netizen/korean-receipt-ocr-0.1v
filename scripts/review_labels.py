"""Local-only label editor. Ctrl+C stops the server."""
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'dataset'
PAGE = '''<!doctype html><meta charset="utf-8"><title>영수증 라벨 검수</title>
<style>body{font:16px sans-serif;margin:20px}main{display:flex;gap:20px}img{max-width:48vw;max-height:85vh;object-fit:contain}textarea{width:45vw;height:72vh}button,select{padding:10px}</style>
<h1>영수증 라벨 검수</h1><p>이미지와 OCR을 대조하고 JSON을 수정하세요. 검수 완료 시 annotation_status를 reviewed로 변경하세요. 개인정보 검수 상태와 구조화 필드는 별도로 기록하세요.</p>
<select id="samples"></select> <button id="save">저장</button> <span id="status"></span>
<main><img id="image"><textarea id="label" spellcheck="false"></textarea></main>
<script>
let records=[];const s=document.querySelector('#samples'),t=document.querySelector('#label');
async function load(){const r=records[s.selectedIndex];document.querySelector('#image').src=r.image;t.value=JSON.stringify(await(await fetch(r.label)).json(),null,2);document.querySelector('#status').textContent='';}
fetch('manifest.json').then(r=>r.json()).then(r=>{records=r.sort((a,b)=>(a.display_name||a.id).localeCompare(b.display_name||b.id));records.forEach(x=>s.add(new Option(x.display_name||x.image.split('/').pop(),x.id)));load()});s.onchange=load;
document.querySelector('#save').onclick=async()=>{try{const value=JSON.parse(t.value);const r=await fetch('/save/'+s.value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});if(!r.ok)throw Error(await r.text());document.querySelector('#status').textContent='저장됨 — JSONL 반영: python scripts/prepare_dataset.py --skip-ocr';}catch(e){alert(e.message)}};
</script>'''

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(ROOT), **kwargs)
    def do_GET(self):
        if self.path == '/':
            data=PAGE.encode('utf-8'); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(data)
        else: super().do_GET()
    def do_POST(self):
        allowed={r['id']:r for r in json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))}
        key=self.path.removeprefix('/save/')
        if not self.path.startswith('/save/') or key not in allowed:
            self.send_error(404);return
        try:
            label=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if label['id'] != key: raise ValueError('id must match the selected image')
            if label['annotation_status'] not in {'pending','auto_labeled_needs_review','reviewed','rejected'}: raise ValueError('Invalid annotation_status')
            path=ROOT/allowed[key]['label']
            path.with_suffix('.json.bak').write_bytes(path.read_bytes())
            path.write_text(json.dumps(label,ensure_ascii=False,indent=2),encoding='utf-8')
            self.send_response(200);self.end_headers();self.wfile.write(b'OK')
        except (ValueError,KeyError) as e: self.send_error(400,str(e))

if __name__ == '__main__':
    print('Open http://127.0.0.1:8765 — Ctrl+C to stop')
    HTTPServer(('127.0.0.1',8765),Handler).serve_forever()
