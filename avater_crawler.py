from glob import glob
import random
from certifi import contents
import requests
from bs4 import BeautifulSoup as Soup
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
from uuid import uuid4
import face_alignment
import math
from fake_useragent import UserAgent
from db import CrawlerDB
from tqdm import tqdm


def download_img(url: str, save_dir: str, count, ua: UserAgent, session):
    user_agent = ua.random
    if not url.startswith('https://images'):
        return
    # content = requests.get(
    #     url, headers={'User-Agent': user_agent}).content
    content = session.get(url).content
    image = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        return
    cv2.imwrite(f'{save_dir}/{count:06}.png', image)
    return True


def browser_init():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    s = Service("/home/yuan/桌面/tempcode/crawler_ettody/chromedriver")
    browser = webdriver.Chrome(
        service=s, options=options)
    return browser


def crawler(browser, crawler_keyword, max_count, save_dir):
    db = CrawlerDB('CrawlerDB.db')
    finished = False
    crawler_keyword = crawler_keyword.replace(' ', '-')
    url = f'https://unsplash.com/s/photos/{crawler_keyword}'
    browser.get(url)
    ua = UserAgent()
    session = requests.Session()
    session.headers['user-agent'] = ua.random
    count = 0
    img_list = os.listdir(save_dir)
    if img_list:
        count = len(img_list)
    max_count += count
    print(crawler_keyword, count, max_count)
    # return
    while not finished:

        soup = Soup(browser.page_source, 'lxml')
        button_content = soup.find_all(
            'button', class_='CwMIr DQBsa p1cWU jpBZ0 AYOsT Olora I0aPD dEcXu')
        if button_content:
            print("button is exists")
            button = browser.find_element(
                By.XPATH, "//button[@class='CwMIr DQBsa p1cWU jpBZ0 AYOsT Olora I0aPD dEcXu']")
            button.click()
            time.sleep(2)
        results = soup.find_all(class_='YVj9w')
        for result in results:
            img_url = result.get('src')
            if db.check_exists(img_url) or img_url is None:
                continue
            download = download_img(img_url, save_dir, count, ua, session)
            if download:
                count += 1
            db.insert_data(crawler_keyword, img_url)
            if count > max_count:
                finished = True
                break
        # browser.execute_script('window.scrollBy(0,1500)')
        browser.execute_script(
            "window.scrollBy(0, document.body.scrollHeight);")
        browser.execute_script('window.scrollBy(0,-1500)')
        time.sleep(random.randrange(1, 3))


def main_crawler(download_dir, crawler_keyword):
    save_dir = os.path.join(download_dir, crawler_keyword)
    max_count = 10000
    os.makedirs(save_dir, exist_ok=True)
    browser = browser_init()
    crawler(browser, crawler_keyword, max_count, save_dir)
    # browser.close()
    browser.quit()


def fa_init():
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, face_detector='blazeface', device='cuda')
    return fa


def preprocess_crop(in_dir, crop_dir_root):
    fa = fa_init()
    # crop_dir = 'test_crop'
    crop_dir = os.path.join(crop_dir_root, os.path.basename(in_dir))
    if os.path.exists(crop_dir):
        return
    print('start crop', crop_dir)
    os.makedirs(crop_dir, exist_ok=True)
    for path in tqdm(sorted(glob(f'{in_dir}/*g'))):
        img = cv2.imread(path)
        max_h, max_w = img.shape[:2]
        # print(max_h, max_w)
        result = fa.get_landmarks(path, return_bboxes=True)
        if not result:
            continue
        _, bboxes = result
        bbox, score = bboxes[0][:-1], bboxes[0][-1]
        if score < 0.85:
            continue
        bbox_w, bbox_h = bbox[2:]-bbox[:2]
        face_area = bbox_w * bbox_h
        if face_area < 90000:
            continue
        if bbox_h > bbox_w:
            diff = (abs(bbox_h-bbox_w)//2)
            bbox[0] -= diff
            bbox[2] += diff
        if bbox_w > bbox_h:
            diff = (abs(bbox_h-bbox_w)//2)
            bbox[1] -= diff
            bbox[3] += diff
        bbox[0] = math.floor(max(0, bbox[0]-bbox_h*0.2))
        bbox[1] = math.floor(max(0, bbox[1]-bbox_h*0.2))
        bbox[2] = math.ceil(min(max_w-1, bbox[2]+bbox_h*0.2))
        bbox[3] = math.ceil(min(max_h-1, bbox[3]+bbox_h*0.2))
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f'{crop_dir}/{os.path.basename(path)}', crop_img)
        # print('ok', score, os.path.basename(path))


if __name__ == "__main__":

    # main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'female')
    # main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'asian')
    # main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'human')
    # main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'beauty girl')
    # main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'handsome guy')
    main_crawler('/home/yuan/hdd8t/unsplash_dataset/raw', 'face')

    # root = '/home/yuan/hdd8t/unsplash_dataset/raw/'
    # for dir in glob(f'{root}/*'):
    #     preprocess_crop(dir,
    #                     '/home/yuan/hdd8t/unsplash_dataset/crop')

    # preprocess_crop('/home/yuan/hdd8t/unsplash_dataset/raw/human',
    #                 '/home/yuan/hdd8t/unsplash_dataset/crop')
    # preprocess_crop('/home/yuan/hdd8t/unsplash_dataset/raw/beauty girl',
    #                 '/home/yuan/hdd8t/unsplash_dataset/crop')
    # preprocess_crop('/home/yuan/hdd8t/unsplash_dataset/raw/beauty girl',
    #                 '/home/yuan/hdd8t/unsplash_dataset/handsome guy')
