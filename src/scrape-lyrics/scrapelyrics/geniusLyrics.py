import urllib2
import requests
import json
import bs4
import sys

token = '2PtHXnbMH-EAIrKUGitGgN2Edg1lmsLwjvU0jDaETseU8vCPdnxhV0AyslIW_vxD'
user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:51.0) Gecko/20100101 Firefox/51.0"

def genius_get_artists(artist_name):
  """
  given an artist name as a string. 
  returns the top artist in the search results for that string.
  
  """
  URL = 'https://api.genius.com/search?q=' + artist_name.replace(' ','%20') + '&per_page=10'
  request = urllib2.Request(URL)
  request.add_header("Authorization", "Bearer " + token)
  request.add_header("User-Agent", user_agent_string) #otherwise 403 returned  
  response = urllib2.urlopen(request, timeout=4)
  raw = response.read()
  json_obj = json.loads(raw)
  for hit in json_obj["response"]["hits"]:
    primary= hit['result']['primary_artist']
    return (primary["id"],primary["name"])
  
def genius_get_songs(artist_id):
  """
  given and artist id, compile a list of songs as tuples (title, song-url)
  """
  artist_id=str(artist_id)
  result=[]
  baseURL = 'https://api.genius.com/artists/' + artist_id + '/songs'
  i=1
  while i<15:
    URL = baseURL+'?per_page=50&page='+str(i)
    request = urllib2.Request(URL) 
    request.add_header("Authorization", "Bearer " + token)
    request.add_header("User-Agent", user_agent_string) #otherwise 403 returned      
    print 'requesting page ', i
    response = urllib2.urlopen(request, timeout=4)
    raw = response.read()
    json_obj = json.loads(raw)
    for song in json_obj['response']['songs']:
      result.append((song['title'], song['url']))
    next_i = json_obj["response"]["next_page"]
    print 'next page = ', next_i    
    if next_i:
      i+=1
    else:
      print 'done ',len(result), 'songs found'
      break
  return result

def genius_get_lyrics(song_url):
  """
  given a URL of a song, scrape the lyrics and return them as a unicode string
  """
  URL=str(song_url)
  response = requests.get(URL, headers={'User-Agent': user_agent_string})
  soup=bs4.BeautifulSoup(response.text,"lxml")
  return soup.find('lyrics').text.strip()
  

def genius_get_all_lyrics_for_artist(artist):
  """
  given an artist, find the poper spelling, then compile a list of songs, then save lyrics to a file and report the progress
  """
  art = genius_get_artists(artist)
  song_list = genius_get_songs(art[0])
  lyricFile = open('/home/thom/projects/Lyrics/'+art[1]+'Lyrics.txt','wb')
  count=1.0
  total=float(len(song_list))
  print "get lyrics for artist "+art[1]
  for song in song_list:
    progress=100*(count/total)
    if progress%10.0<0.5:
      print progress,"%"
    surl=song[1]
    l=u"\nTitle:"+song[0]+u"\n"+genius_get_lyrics(surl)
    lyricFile.write(l.encode('utf-8','ignore'))  
    count+=1
  lyricFile.close()  
  
  
def main():  
  desired=['led zeppelin','pink floyd']
  for a in desired:
    print "compile songs for artist: "+a
    genius_get_all_lyrics_for_artist(a)
    
if __name__=='__main__':
  main()
