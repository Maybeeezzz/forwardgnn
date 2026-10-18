"""Read one debug event and its variable/storage inventory without PyTorch."""
import argparse
import json
from pathlib import Path


def find_event(path, sequence):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sequence'] == sequence:
                return row
    raise ValueError('Event {} not found in {}'.format(sequence, path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='One SF/BP worker output directory')
    parser.add_argument('--sequence', type=int, help='Default: highest observed boundary allocation')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    sequence = args.sequence
    if sequence is None:
        with (args.directory / 'memory-events.jsonl').open() as stream:
            peak = max((json.loads(line) for line in stream), key=lambda row: row['allocated_bytes'])
        sequence = peak['sequence']
    event = find_event(args.directory / 'memory-events.jsonl', sequence)
    details = find_event(args.directory / 'memory-variables.jsonl', sequence)
    print('Event {}\nInterval after: {}\nBoundary: {}'.format(sequence, event['interval_after'], event['end']))
    for label in ('allocated_bytes', 'reserved_bytes', 'interval_peak_allocated_bytes',
                  'allocated_delta_bytes', 'peak_to_end_drop_bytes',
                  'visible_unique_storage_bytes', 'allocator_minus_visible_storage_bytes'):
        if label in event:
            print('{}: {:.4f} MiB'.format(label, event[label] / 2**20))
    print('\nLargest visible storages (deduplicated):')
    for entry in sorted(details['storages'], key=lambda row: row['bytes'], reverse=True)[:args.top]:
        print('{:.4f} MiB  {}'.format(entry['bytes'] / 2**20, entry['storage_id']))
        for alias in entry['aliases']:
            print('  ' + alias)
    print('\nVariables (logical size; aliases may share storage):')
    for row in sorted(details['variables'], key=lambda row: row['logical_bytes'], reverse=True)[:args.top]:
        print('{} shape={} {} logical={:.4f} MiB storage={} offset={}'.format(
            row['path'], row['shape'], row['dtype'], row['logical_bytes'] / 2**20,
            row['storage_id'], row['storage_offset']))
    for field in ('appeared_storages', 'disappeared_storages'):
        print('\n{} (visibility changes, NOT exact allocator alloc/free):'.format(field))
        for entry in sorted(details[field], key=lambda row: row['bytes'], reverse=True)[:args.top]:
            print('{:.4f} MiB  {}'.format(entry['bytes'] / 2**20, '; '.join(entry['aliases'])))
    if details['inspection_errors']:
        print('\nIncomplete inspection:', details['inspection_errors'])


if __name__ == '__main__':
    main()
