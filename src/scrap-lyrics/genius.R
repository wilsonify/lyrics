library(tidyverse)
library(httr)
library(stringr)
library(rvest)

token <- '2PtHXnbMH-EAIrKUGitGgN2Edg1lmsLwjvU0jDaETseU8vCPdnxhV0AyslIW_vxD'
GET
genius_get_artists <- function(artist_name, n_results = 10) {
  baseURL <- 'https://api.genius.com/search?q=' 
  requestURL <- paste0(baseURL, gsub(' ', '%20', artist_name), '&per_page=', n_results, '&access_token=', token) #paste0=Concatenate vectors after converting to character.
  res <- GET(requestURL) %>% content %>% .$response %>% .$hits
  View(res)
  map_df(1:length(res), function(x) {
    tmp <- res[[x]]$result$primary_artist
    list( artist_id = tmp$id, artist_name = tmp$name )
  }) %>% unique
}

lyric_scraper <- function(url) {
    read_html(url) %>% 
    html_node('lyrics') %>% 
    html_text
}

genius_artists <- genius_get_artists('chvrches')
genius_artists

  baseURL <- 'https://api.genius.com/artists/' 
  requestURL <- paste0(baseURL, genius_artists$artist_id[1], '/songs')

  track_lyric_urls <- list()

  i <- 1
  while (i > 0) {
    tmp <- GET(requestURL, query = list(access_token = token, per_page = 50, page = i)) %>% content %>% .$response
    track_lyric_urls <- c(track_lyric_urls, tmp$songs)
    if (!is.null(tmp$next_page)) {
      i <- tmp$next_page
      } else {
        break
      }
  }

track_lyric_urls
  
genius_df <- map_df(1:length(track_lyric_urls), function(x) {
  lyrics <- lyric_scraper(track_lyric_urls[[x]]$url)
  tots <- list(track_name = track_lyric_urls[[x]]$title, lyrics = lyrics)
  return(tots)
})

write.csv(genius_df, file="chvrchesLyrics.csv")