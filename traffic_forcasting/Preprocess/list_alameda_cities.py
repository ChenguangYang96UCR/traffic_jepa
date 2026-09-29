#!/usr/bin/env python3
"""List non-empty cities and sensor counts from Alameda sensors.csv."""
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import re
import unicodedata


def find_city_column(columns):
    normalized = {
        ''.join(character for character in str(column).lower()
                if character.isalnum()): column
        for column in columns
    }
    if 'city' not in normalized:
        raise ValueError(f'No City column; available columns: {columns}')
    return normalized['city']


def slugify(city):
    ascii_city = unicodedata.normalize('NFKD', city).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '_', ascii_city.casefold()).strip('_')
    return slug or f'city_{hashlib.sha256(city.encode()).hexdigest()[:8]}'


def city_specs(path, requested=''):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        city_column = find_city_column(columns)
        values = [str(row.get(city_column, '')).strip() for row in reader]
    counts = Counter(value.casefold() for value in values if value)
    by_fold = {}
    for value in values:
        if value:
            by_fold.setdefault(value.casefold(), value)
    selected = list(dict.fromkeys(
        value.strip().casefold() for value in requested.split(',') if value.strip()))
    if selected:
        missing = sorted(set(selected) - set(by_fold))
        if missing:
            raise ValueError(
                f'Requested cities are absent: {missing}; available={sorted(by_fold.values())}')
        city_keys = selected
    else:
        city_keys = sorted(counts)

    seen = {}
    result = []
    for city_key in city_keys:
        city = by_fold[city_key]
        slug = slugify(city)
        if slug in seen and seen[slug] != city:
            slug = f'{slug}_{hashlib.sha256(city.encode()).hexdigest()[:8]}'
        seen[slug] = city
        result.append((city, slug, counts[city_key]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sensors_csv', type=Path)
    parser.add_argument('--cities', default='', help='optional comma-separated subset')
    args = parser.parse_args()
    for city, slug, count in city_specs(args.sensors_csv, args.cities):
        print(f'{city}\t{slug}\t{count}')


if __name__ == '__main__':
    main()
