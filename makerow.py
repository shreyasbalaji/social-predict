import os
import numpy as np
# import tensorflow as tf
import json
import re
from multiprocessing import Pool
from collections import namedtuple
from PIL import Image
import json
from datetime import datetime
import requests
import subprocess


MAX_SIZE = 500
root_dir = os.path.dirname(__file__)
SerializeInfo = namedtuple('SerializeInfo', ['input_file', 'input_json', 'output_prefix'])


def serialize_image(info):
    try:
        img = Image.open(info.input_file)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img.save(f'{info.output_prefix}.jpg', "JPEG")
        os.rename(info.input_json, f'{info.output_prefix}.json')
        os.remove(info.input_file)
    except IOError as e:
        print(e)
        print("Could not resize image")
        print(info.input_json)
        print(f'{info.output_prefix}.json')
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
        if fname[-4:] == '.mp4':
            os.remove(os.path.join(input_dir, fname))
        elif fname[-4:] == '.jpg':
            test_search = re.search('\d+.jpg', fname)
            if test_search:
                os.remove(os.path.join(input_dir, fname))
                continue
            else:
                fname_prefix = fname[:-4]

            # print(fname)
            with open(os.path.join(odir, 'meta.json'), 'r') as f:
                csize = json.load(f)['size']
            if csize >= MAX_SIZE:
                current_id = int(odir[-4:].lstrip('0'))
                fid = "%04d" % (current_id + 1)
                odir = os.path.join(output_dir, f'tff{fid}')
                if not os.path.exists(odir):
                    os.makedirs(odir)
                if not os.path.exists(os.path.join(odir, 'meta.json')):
                    with open(os.path.join(odir, 'meta.json'), 'w') as f:
                        json.dump({'size': 0}, f)
                with open(os.path.join(odir, 'meta.json')) as f:
                    csize = json.load(f)['size']

            csize += 1
            jpg_prefix = "%04d" % csize
            infos.append(SerializeInfo(
                os.path.join(input_dir, fname),
                os.path.join(input_dir, f'{fname_prefix}.json'),
                os.path.join(odir, f'{jpg_prefix}')))

            with open(os.path.join(odir, 'meta.json'), 'w') as f:
                json.dump({'size': csize}, f)

    pool = Pool(4)
    pool.map(serialize_image, infos)

serialize_images(os.path.join(root_dir, 'selfies'), os.path.join(root_dir, 'data'))

def process_json(filename):
    with open(filename) as f:
        data = json.load(f)
    caption = data["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
    day, hour = convert_time(data["node"]["taken_at_timestamp"])
    liked_by = data["node"]["edge_liked_by"]["count"]
    followers = get_followers(data["node"]["owner"]["id"])
    return [caption, day, hour, liked_by, followers]


def convert_time(time):
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    week_day = datetime.utcfromtimestamp(time).strftime('%A')
    hour_day = int(datetime.utcfromtimestamp(time).strftime('%H'))

    vector_day = []
    for i in days:
        if i == week_day:
            vector_day.append(1)
        else:
            vector_day.append(0)

    return vector_day, hour_day


def get_followers(user_id):
    cmd = "curl -s https://i.instagram.com/api/v1/users/" + user_id + "/info/ | jq -r .user.username"
    username = subprocess.check_output(cmd, shell=True).decode("utf-8")[:-1]
    follower_count = requests.get("https://www.instagram.com/web/search/topsearch/?query={" + username + "}").json()["users"][0]["user"]["follower_count"]
    return follower_count

# print(process_json("eeifshemaisrael/2017-04-11_22-21-37_UTC.json"))
