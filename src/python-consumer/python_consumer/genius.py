import requests
import json
import bs4
import os
from collections import defaultdict

TOKEN = "2PtHXnbMH-EAIrKUGitGgN2Edg1lmsLwjvU0jDaETseU8vCPdnxhV0AyslIW_vxD"


def genius_get_artists(artist_name):
    """
    given an artist name as a string.
    returns the most frequent artist in the first 10 genius search results for that string.
    """
    try:
        URL = (
                "https://api.genius.com/search?q="
                + artist_name.replace(" ", "%20")
                + "&per_page=10"
        )
        raw = requests.get(URL, headers={"Authorization": "Bearer " + TOKEN})
        json_obj = json.loads(raw.content)
        result = defaultdict(int)
        for hit in json_obj["response"]["hits"]:
            hit = hit["result"]["primary_artist"]
            result[(hit["id"], hit["name"])] += 1
        return max(iter(result.items()), key=lambda x: x[1])[0]
    except:
        print("no results found. usage: genius_get_artist(string) return (id, name)")
        raise


def genius_get_songs(artist_id):
    """
    given an artist id, compile a list of songs as tuples (title, url)
    """
    try:
        artist_id = str(artist_id)
        result = []
        baseURL = "https://api.genius.com/artists/" + artist_id + "/songs"
        i = 1
        while i:
            URL = baseURL + "?per_page=50&page=" + str(i)
            raw = requests.get(URL, headers={"Authorization": "Bearer " + TOKEN})
            json_obj = json.loads(raw.content)
            for song in json_obj["response"]["songs"]:
                result.append((song["title"], song["url"]))
            next_i = json_obj["response"]["next_page"]
            print("next page = ", next_i)
            if next_i:
                print(len(result), "found, next page")
                i += 1
            else:
                print("done ", len(result), "songs found")
                break
        return result
    except:
        print(
            "no songs found. usage: genius_get_songs(id) return list of tuples [(title,url)...]"
        )
        raise


def genius_get_lyrics(song_url):
    """
    given a URL of a song, scrape the lyrics and return them as a unicode string
    """
    try:
        response = requests.get(song_url)
        soup = bs4.BeautifulSoup(response.text, "lxml")
        return soup.find("lyrics").text.strip()
    except:
        print(
            "cannot find lyrics. usage: genius_get_lyrics(song_url) returns lyrics as unicode string"
        )
        raise


def genius_get_all_lyrics_for_artist(artist):
    """
    given an artist, find the proper spelling, then compile a list of songs, then save lyrics to a file and report the progress
    """
    art = genius_get_artists(artist)
    song_list = genius_get_songs(art[0])

    try:
        if not os.path.exists("/home/thom/projects/Lyrics/" + art[1]):
            os.makedirs("/home/thom/projects/Lyrics/" + art[1])
        print("get lyrics for artist " + art[1])
        count = 0
        for song in song_list:
            lyricFile = open(
                "/home/thom/projects/Lyrics/"
                + art[1]
                + "/"
                + song[0].replace("/", "_")
                + ".txt",
                "wb",
            )
            surl = song[1]
            l = "\nTitle:" + song[0] + "\n" + genius_get_lyrics(surl)
            lyricFile.write(l.encode("utf-8", "ignore"))
            count += 1
            lyricFile.close()
    except:
        print(
            "cannot get lyrics usage: genius_get_all_lyrics_for_artist(artist) creates one text file per song retrieved from genius.com"
        )
        raise
