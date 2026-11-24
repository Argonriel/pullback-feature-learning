"""Preserve original-source bytes and explicit retrieval failures; no external writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
import hashlib, json, datetime

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ('lesniewski_ruskai.pdf', 'https://arxiv.org/pdf/math-ph/9808016'),
    ('grasselli_streater.pdf', 'https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf'),
    ('fawzi_saunderson_parrilo.pdf', 'https://arxiv.org/pdf/1705.00812'),
    ('bresar_cabrera_fosner_villena.pdf', 'https://www.impan.pl/shop/en/publication/transaction/download/product/90340'),
    ('ohara_ishi_tsuchiya.html', 'https://link.springer.com/article/10.1007/s41884-023-00116-x'),
    ('nagaoka_fujiwara.html', 'https://arxiv.org/html/2307.03431v3'),
    ('ciaglia_dinocera.pdf', 'https://arxiv.org/pdf/2206.10394'),
    ('li_wang_lee_arora.pdf', 'https://papers.nips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Paper-Conference.pdf'),
]

def retrieve(item):
    name, url = item
    row = {'file': 'sources/' + name, 'url': url,
           'retrieved_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=25) as response:
            data = response.read()
            row['content_type'] = response.headers.get('Content-Type')
            row['resolved_url'] = response.geturl()
        if name.endswith('.pdf') and not data.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        (ROOT / row['file']).write_bytes(data)
        row.update(status='saved', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        row.update(status='failed', error=repr(exc))
    return row

if __name__ == '__main__':
    (ROOT / 'sources').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(retrieve, SOURCES))
    out = {'purpose':'Primary source byte provenance; relevant passages independently inspected via web tools.',
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records':records}
    (ROOT / 'SOURCE_HASHES.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('file','status')} for r in records]))
