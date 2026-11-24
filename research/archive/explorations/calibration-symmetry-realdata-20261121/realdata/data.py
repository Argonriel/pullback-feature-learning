"""Official Fashion-MNIST IDX download and guarded access."""
from pathlib import Path
import gzip, hashlib, json, struct, urllib.request
from datetime import datetime, timezone
import numpy as np
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310'}
BASE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expected in FILES.items():
        path = DATA / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
        content = path.read_bytes()
        digest = hashlib.md5(content).hexdigest()
        assert digest == expected, (name, digest)
        rows.append(dict(file=name, url=BASE+name, md5=digest, sha256=sha(path), bytes=len(content)))
        print('Downloaded and checksum-verified', name, flush=True)
    license_path = DATA / 'LICENSE_FASHION_MNIST.txt'
    if not license_path.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE', license_path)
    (DATA / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(dict(utc=now(), files=rows,
        official_source='https://github.com/zalandoresearch/fashion-mnist', license='MIT',
        license_sha256=sha(license_path), test_arrays_decoded=False), indent=2))


def arrays(split):
    if split == 'test':
        approval = HERE / 'TEST_APPROVAL.json'
        assert approval.exists(), 'Official test arrays are locked until root approval.'
        record = json.loads(approval.read_text())
        assert record['test_access_authorized'] is True
        assert record['policy_commit_sha256'] == sha(HERE / 'POLICY_COMMIT.json')
        prefix = 't10k'
    elif split == 'train':
        prefix = 'train'
    else:
        raise ValueError(split)
    im = gzip.decompress((DATA / (prefix + '-images-idx3-ubyte.gz')).read_bytes())
    lab = gzip.decompress((DATA / (prefix + '-labels-idx1-ubyte.gz')).read_bytes())
    magic, count, rows, cols = struct.unpack('>4I', im[:16])
    assert magic == 2051 and (rows, cols) == (28, 28)
    lm, ln = struct.unpack('>2I', lab[:8]); assert lm == 2049 and ln == count
    return np.frombuffer(im, np.uint8, offset=16).reshape(count, 1, 28, 28).copy(), np.frombuffer(lab, np.uint8, offset=8).astype(np.int64)


def splits(labels):
    rng = np.random.default_rng(2131000)
    sections = dict(train=[], validation=[], temperature=[], selector_fit=[], selector_validation=[])
    edges = [0, 4000, 4500, 5000, 5800, 6000]
    for label in range(10):
        ids = np.flatnonzero(labels == label)
        assert len(ids) == 6000
        ids = rng.permutation(ids)
        for j, name in enumerate(sections):
            sections[name].append(ids[edges[j]:edges[j+1]])
    sections = {name: rng.permutation(np.concatenate(values)) for name, values in sections.items()}
    assert len(np.unique(np.concatenate(list(sections.values())))) == 60000
    return sections


if __name__ == '__main__':
    download()
