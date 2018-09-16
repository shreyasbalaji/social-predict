import os
import numpy as np
# import tensorflow as tf
import json
from multiprocessing import Pool
from collections import namedtuple
from PIL import Image


MAX_SIZE = 1000
root_dir = os.path.dirname(__file__)
SerializeInfo = namedtuple('SerializeInfo', ['input_file', 'output_file'])


def serialize_image(info):
    try:
        img = Image.open(info.input_file)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img.save(info.output_file, "JPEG")
    except IOError:
        print("Could not resize image")
    return None


def serialize_images(input_dir, output_dir):
    """ meta filename = 'meta.json' """

    odirs = os.listdir(output_dir)
    odirs = list(filter(lambda d: d[:3] == 'tff', odirs))
    odirs.sort()

    if len(odirs) == 0:
        odir = os.path.join(output_dir, 'tff0001')
        os.makedirs(odir)
    else:
        odir = os.path.join(output_dir, odirs[-1])

    if not os.path.exists(os.path.join(odir, 'meta.json')):
        with open(os.path.join(odir, 'meta.json'), 'w') as f:
            json.dump({'size': 0}, f)

    fnames = list(os.listdir(input_dir))
    infos = []
    for fname in fnames:
        if fname[-4:] == '.jpg':
            # print(fname)
            with open(os.path.join(odir, 'meta.json'), 'r') as f:
                csize = json.load(f)['size']
            if csize >= MAX_SIZE:
                current_id = int(fname[:4].lstrip('0'))
                fid = "%04d" % (current_id + 1)
                odir = os.path.join(output_dir, f'tff{fid}')
                if not os.path.exists(odir):
                    os.makedirs(odir)
                if not os.path.exists(os.path.join(odir, 'meta.json')):
                    with open(os.path.join(odir, 'meta.json'), 'r') as f:
                        json.dump({'size': 0}, f)
                with open(os.path.join(odir, 'meta.json')) as f:
                    csize = json.load(f)['size']

            csize += 1
            jpg_prefix = "%04d" % csize
            infos.append(SerializeInfo(
                os.path.join(input_dir, fname),
                os.path.join(odir, f'{jpg_prefix}.jpg')))


            with open(os.path.join(odir, 'meta.json'), 'w') as f:
                json.dump({'size': csize}, f)

    pool = Pool(4)
    pool.map(serialize_image, infos)

serialize_images(os.path.join(root_dir, 'eeifshemaisrael'), os.path.join(root_dir, 'data'))
