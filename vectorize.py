from datetime import datetime
import json
import requests
import subprocess
import numpy as np


def process_json(filename):
    with open(filename) as f:
        data = json.load(f)
    caption = data["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
    day, hour = convert_time(data["node"]["taken_at_timestamp"], caption)
    liked_by = data["node"]["edge_liked_by"]["count"]
    followers = int(get_followers(data["node"]["owner"]["id"]))
    return get_feature_vector(
            caption,
            day,
            hour,
            followers,
            liked_by
        )


def convert_time(time, caption):
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    hashtags = ['#nycselfie', '#laselfie', '#bostonselfie', '#londonselfie', '#parisselfie', '#caliselfie', '#chicagoselfie']
    timezone_change = [-4, -7, -4, 1, 2, -7, -5]
    week_day = datetime.utcfromtimestamp(time).strftime('%A')
    hour_day = int(datetime.utcfromtimestamp(time).strftime('%H'))

    index_timezone = 0
    for i in range(len(hashtags)):
        if hashtags[i] in caption:
            index_timezone = i
            break

    change_day = (hour_day + timezone_change[index_timezone])//24
    hour_day = (hour_day + timezone_change[index_timezone]) % 24
    vector_day = []
    day_index = (days.index(week_day) + change_day) % 7

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


def get_feature_vector(caption, day, hour, followers, liked_by):
    return np.array([followers, hour, *day, liked_by], dtype=np.float32) 
