import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/'
REVIEW_FILE = 'goodreads_reviews_fantasy_paranormal.json.gz'
BOOK_FILE = 'goodreads_books_fantasy_paranormal.json.gz'
ALIASES = {REVIEW_FILE: 'reviews.json.gz', BOOK_FILE: 'books.json.gz'}


def download(filename, folder='data'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in (ALIASES.get(filename, filename), filename):
        path = folder / name
        if path.is_file() and path.stat().st_size > 100:
            with path.open('rb') as stream:
                if stream.read(2) != b'\x1f\x8b':
                    raise ValueError(f'Not a gzip file: {path}')
            return path
    destination = folder / filename
    temporary = destination.with_name(destination.name + '.part')
    try:
        with requests.get(BASE_URL + filename, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
        with gzip.open(temporary, 'rb') as stream:
            if not stream.read(1):
                raise ValueError(f'Empty Goodreads archive: {filename}')
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_jsonl_gz(path, limit=None):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def kcore(frame, min_user=5, min_item=10):
    while True:
        old_size = len(frame)
        frame = frame[frame.user_id.map(frame.user_id.value_counts()).ge(min_user)]
        frame = frame[frame.book_id.map(frame.book_id.value_counts()).ge(min_item)]
        if len(frame) == old_size:
            return frame


def prepare_data(folder='data', scan_limit=900_000, max_items=2500,
                 max_users=12000, max_events=180_000, min_user=5, min_item=10,
                 min_events=100_000, min_users=1000, min_items=1000):
    records = []
    for row in read_jsonl_gz(download(REVIEW_FILE, folder), scan_limit):
        if 'user_id' not in row:
            raise ValueError(
                'Reviews do not contain user_id. The goodreads_sample.zip supplied '
                'for debugging cannot be used to train recommenders. Use the '
                'original UCSD goodreads_reviews_fantasy_paranormal.json.gz '
                'or reviews.json.gz retaining user_id.'
            )
        try:
            rating = int(row['rating'])
            if 1 <= rating <= 5:
                records.append((
                    str(row['user_id']), str(row['book_id']), rating,
                    row.get('date_added') or row.get('date_updated'),
                ))
        except (KeyError, TypeError, ValueError):
            continue
    if not records:
        raise ValueError('No valid reviews were read. Check the input archive.')
    frame = pd.DataFrame(records, columns=['user_id', 'book_id', 'rating', 'timestamp'])
    frame['timestamp'] = pd.to_datetime(
        frame.timestamp, format='%a %b %d %H:%M:%S %z %Y',
        utc=True, errors='coerce',
    )
    frame = frame.dropna(subset=['timestamp'])
    frame = frame.sort_values(['timestamp', 'user_id', 'book_id'])
    frame = frame.drop_duplicates(['user_id', 'book_id'], keep='last')
    top_items = frame.book_id.value_counts().head(max_items).index
    frame = frame[frame.book_id.isin(top_items)]
    top_users = frame.user_id.value_counts().head(max_users).index
    frame = frame[frame.user_id.isin(top_users)]
    frame = kcore(frame, min_user, min_item)
    if len(frame) > max_events:
        hashes = pd.util.hash_pandas_object(frame[['user_id', 'book_id']], index=False)
        frame = frame.loc[hashes.nsmallest(max_events).index]
        frame = kcore(frame, min_user, min_item)
    frame = frame.sort_values(['timestamp', 'user_id', 'book_id']).reset_index(drop=True)
    users_count = frame.user_id.nunique()
    items_count = frame.book_id.nunique()
    if len(frame) < min_events or users_count < min_users or items_count < min_items:
        raise ValueError(
            f'Sample too small: {len(frame)} events, {users_count} users, '
            f'{items_count} items. Increase scan_limit or sampling limits.'
        )
    needed = set(frame.book_id)
    metadata = {}
    for row in read_jsonl_gz(download(BOOK_FILE, folder)):
        item = str(row.get('book_id', ''))
        if item not in needed:
            continue
        authors = row.get('authors') or []
        author = str(authors[0].get('author_id') or 'unknown') if authors else 'unknown'
        metadata[item] = {
            'title': row.get('title') or '',
            'author': author,
            'language': row.get('language_code') or 'unknown',
            'year': pd.to_numeric(row.get('publication_year'), errors='coerce'),
        }
        if len(metadata) == len(needed):
            break
    meta = pd.DataFrame.from_dict(metadata, orient='index')
    meta = meta.reindex(sorted(needed), columns=['title', 'author', 'language', 'year'])
    for field, default in [('title', ''), ('author', 'unknown'), ('language', 'unknown')]:
        meta[field] = meta[field].fillna(default)
        meta[field] = meta[field].replace('', default)
    meta['year'] = pd.to_numeric(meta.year, errors='coerce')
    meta.loc[~meta.year.between(1000, 2026), 'year'] = np.nan
    meta.index.name = 'book_id'
    return frame, meta


def encode_and_split(frame):
    frame = frame.sort_values(['user_id', 'timestamp', 'book_id']).copy()
    users = sorted(frame.user_id.unique())
    items = sorted(frame.book_id.unique())
    user_map = {value: index for index, value in enumerate(users)}
    item_map = {value: index for index, value in enumerate(items)}
    frame['u'] = frame.user_id.map(user_map).astype(int)
    frame['i'] = frame.book_id.map(item_map).astype(int)
    frame = frame.sort_values(['u', 'timestamp', 'i'])
    order = frame.groupby('u').cumcount(ascending=False)
    train = frame[order >= 2].copy()
    val = frame[order == 1].copy()
    test = frame[order == 0].copy()
    known_items = set(train.i)
    val = val[val.i.isin(known_items)].copy()
    test = test[test.i.isin(known_items)].copy()
    return train, val, test, user_map, item_map


def matrix(frame, n_users, n_items, binary=False):
    values = (np.ones(len(frame), dtype=np.float32) if binary
              else frame.rating.to_numpy(dtype=np.float32))
    return sparse.csr_matrix(
        (values, (frame.u.to_numpy(), frame.i.to_numpy())),
        shape=(n_users, n_items),
    )


class FMFeatures:
    def __init__(self, item_map, metadata, n_users, include_metadata=True):
        self.n_users = n_users
        self.n_items = len(item_map)
        self.include_metadata = include_metadata
        if include_metadata:
            self.vectorizer = DictVectorizer(sparse=True)
            ids = [item for item, _ in sorted(
                item_map.items(), key=lambda entry: entry[1]
            )]
            rows = []
            for item in ids:
                info = metadata.loc[item]
                year = info['year']
                decade = str(int(year // 10 * 10)) if pd.notna(year) else 'unknown'
                rows.append({
                    'author=' + str(info['author']): 1,
                    'language=' + str(info['language']): 1,
                    'decade=' + decade: 1,
                })
            self.item_extra = self.vectorizer.fit_transform(rows).tocsr()
        else:
            self.item_extra = sparse.csr_matrix((self.n_items, 0))
        self.n_features = n_users + self.n_items + self.item_extra.shape[1]

    def transform(self, users, items):
        users = np.asarray(users, dtype=int)
        items = np.asarray(items, dtype=int)
        if users.shape != items.shape:
            raise ValueError('Users and items must have matching shapes')
        count = len(users)
        rows = np.arange(count)
        user_part = sparse.csr_matrix(
            (np.ones(count, dtype=np.float32), (rows, users)),
            shape=(count, self.n_users),
        )
        item_part = sparse.csr_matrix(
            (np.ones(count, dtype=np.float32), (rows, items)),
            shape=(count, self.n_items),
        )
        return sparse.hstack([user_part, item_part, self.item_extra[items]], format='csr')
