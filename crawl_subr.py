import requests
import praw
from imgurpython import ImgurClient
import re
import time
import os
import sys

if __name__ == "__main__":
    found_url = {}
    subr = sys.argv[1]
    if not os.path.exists(os.getcwd()+'/nsfw'):
        os.mkdir('nsfw')
    if not os.path.exists(os.getcwd()+'/normal'):
        os.mkdir('normal')
    # Insert your Imgur and Mashape keys/secrets here
    img_client_id = ''
    img_client_secret = ''
    mashape_key = ''
    if mashape_key:
        img_client = ImgurClient(img_client_id, img_client_secret, mashape_key=mashape_key)
    else:
        img_client = ImgurClient(img_client_id, img_client_secret)
    for i in range(1,25):
        data = {}
        for gallery_img in img_client.subreddit_gallery(subr, page=i):
            url = gallery_img.link
            # check if image is in jpg format and has not already been processed
            if len(re.findall('imgur.com/.+\.jpg',url))>0 and found_url.get(url,0)==0:
                print("writing " + url)
                img_id = re.findall('.com.+\.jpg',url)[0][5:-4]
                try:
                    img = img_client.get_image(img_id)
                except:
                    continue
                # get images in m format to make it easier to process for the network
                response = requests.get(url[:-4]+'m.jpg')
                label = img.nsfw
                if label:
                    path = 'nsfw/'
                else:
                    path = 'normal/'
                try:
                    if not os.path.exists(path+img_id+'.jpg'):
                        f = open(path+img_id+'.jpg',"wb")
                        f.write(response.content)
                        f.close()
                        found_url[url] = 1
                    else:
                        found_url[url] = 1
                except:
                    print(" could not write " + path+img_id+'.jpg')
                    continue
                data[url] = label
