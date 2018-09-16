from datetime import datetime
import json

def process_json(filename):
    with open(filename) as f:
        data = json.load(f)
    caption = data["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
    day, hour = convert_time(data["node"]["taken_at_timestamp"])
    liked_by = data["node"]["edge_liked_by"]["count"]
    followers = int(get_followers(data["node"]["owner"]["id"]))
    return get_feature_vector(
            caption,
            day,
            hour,
            followers,
            liked_by
        )
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

def get_feature_vector(elements):


# print(process_json("eeifshemaisrael/2017-04-11_22-21-37_UTC.json"))
