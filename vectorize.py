from datetime import datetime
import json
import requests
import subprocess
import numpy as np
import spacy

nlp = spacy.load('en')

def process_json(filename):
    with open(filename) as f:
        data = json.load(f)
    try:
        caption = data["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
    except Exception:
        caption = ""
    try:
        day, hour = convert_time(data["node"]["taken_at_timestamp"], caption)
    except Exception:
        day, hour = convert_time(None, caption, fake=True)
    try:
        liked_by = data["node"]["edge_liked_by"]["count"]
    except Exception:
        liked_by = 100
    try:
        followers = int(get_followers(data["node"]["owner"]["id"]))
    except Exception:
        followers = 100
    return get_feature_vector(
            caption,
            day,
            hour/24,
            followers/250,
            liked_by/(followers+1)
        )


def convert_time(time, caption, fake=False):
    if fake:
        day_index = 0
        hour_day = 12
    else:
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        hashtags = ['#nycselfie', '#laselfie', '#bostonselfie', '#londonselfie', '#parisselfie', '#caliselfie',
                     '#chicagoselfie']
        timezone_change = [-4, -7, -4, 1, 2, -7, -5]
        week_day = datetime.utcfromtimestamp(time).strftime('%A')
        hour_day = int(datetime.utcfromtimestamp(time).strftime('%H'))

        timezone_index = 0
        for i in range(len(hashtags)):
            if hashtags[i] in caption:
                timezone_index = i
                break
        change_day = (hour_day + timezone_change[timezone_index]) // 24
        hour_day = (hour_day + timezone_change[timezone_index]) % 24
        day_index = (change_day + days.index(week_day)) % 7

    vector_day = []

    for i in range(7):
        if i == day_index:
            vector_day.append(1)
        else:
            vector_day.append(0)

    return vector_day, hour_day


def get_followers(user_id):
    cmd = "curl -s https://i.instagram.com/api/v1/users/" + user_id + "/info/ | jq -r .user.username"
    username = subprocess.check_output(cmd, shell=True).decode("utf-8")[:-1]
    follower_count = requests.get("https://www.instagram.com/web/search/topsearch/?query={" + username + "}").json()["users"][0]["user"]["follower_count"]
    return follower_count


def get_feature_vector(caption, day, hour, followers, liked_by_perc):
    return np.concatenate((
            nlp('caption ' + str(caption)).vector,
            np.array([followers, hour, *day, liked_by_perc], dtype=np.float32)
        ))

print(process_json("eeifshemaisrael/2017-04-11_22-21-37_UTC.json"))
