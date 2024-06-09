from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
import os

regex_identifier = re.compile('<[^<>]+>([^<>]+)</a>')
regex_group = re.compile('([^-]+)-[^-]+-[^-]+')
base_url = 'https://pool.publicdomainproject.org/audio/flac/'

def build_link(tag):
    #example for a tag: <a href="/index.php/Africavox-dg1-ax98" title="Africavox-dg1-ax98">Africavox-dg1-ax98</a>
    #the identifier from this is Africavox-dg1-ax98
    identifier = regex_identifier.match(tag).groups()[0]

    #In some of the links the identifier looks a bit different (ex: Ace of Diamonds GOS 585/7) but the number of these are small, so I just ignore them 
    try:
        group = regex_group.match(identifier).groups()[0]
    except(AttributeError):
        return None
    return f'{base_url}{group.lower()}/{identifier.lower()}.flac'

def main():
    # Uncomment this if you dont have the   links_to_audio.txt file. This commented section creates that text file collecting the links tro individual audio files from the Publuic Domain Project

    # links_to_audio = []
    # page_urls = [
    #     'https://web.archive.org/web/20230609103058/http://pool.publicdomainproject.org/index.php/Category:FLAC_sound_files',
    #     'https://web.archive.org/web/20220117105221/https://pool.publicdomainproject.org/index.php?title=Category:FLAC_sound_files&pagefrom=Dgg-68132b-2364ge5#mw-pages'
    # ]

    # for url in page_urls:
    #     page = requests.get(url)

    #     soup = BeautifulSoup(page.text, features="html.parser")
    #     soundfile_tags = soup.select('div.mw-category-group ul li a')
    #     print(len(soundfile_tags))
    #     for soundfile_tag in soundfile_tags:
    #         audio_link = build_link(str(soundfile_tag))
    #         if audio_link:
    #             links_to_audio.append(audio_link)
    # print(len(links_to_audio))
    # with open('data/links_to_audio.txt', 'w') as file:
    #     for link in links_to_audio:
            # file.write(f'{link}\n')


    # The following code sectuion loads in the links from links_to_audio.txt
    links_to_audio = []

    if not os.path.exists('data/noisy_audio'):
        os.makedirs('data/noisy_audio')

    with open('data/links_to_audio.txt', 'r') as file:
        links_to_audio = file.readlines()

    counter = 0
    for i in tqdm(range(0,10)):
        link = links_to_audio[i]
        response = requests.get(link[:-1])
        try:
            response.raise_for_status()
            path = f'data/noisy_audio/noisy_audio_{counter}.flac'
            with open(path, 'wb') as soundfile:
                soundfile.write(response.content)
                counter += 1
        except requests.exceptions.HTTPError:
            pass

if __name__ == '__main__':
    main()