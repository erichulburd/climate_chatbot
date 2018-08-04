from trainer import storage
import argparse
import json
import os

if __name__ == '__main__':
    storage.set_bucket(os.getenv('BUCKET'))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        help='Base file path',
        default=''
    )
    args = parser.parse_args()
    arguments = args.__dict__
    hypes = json.load(open('hypes.json'))
    storage.write(json.dumps(hypes), '%s/hypes.json' % arguments['path'], content_type='application/json')
