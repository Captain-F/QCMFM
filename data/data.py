from bs4 import BeautifulSoup
from selenium import webdriver
import urllib
import osa
import time
import joblib


def read_pickle(path):
    with open(path, 'rb') as f:
        feats = joblib.load(f)

    return feats


def save_pickle(path, feats):
    with open(path, 'wb') as f:
        joblib.dump(feats, f)


# 打开目标网页
categ_id_name = {"categories_dict"}

for categ in ["categories"]:

    print(f"******{categ_id_name[categ]}******")

    # for page in range(1, 19):
    for page in range(1, 19):
        print("***第%d页***" % page)

        url = f"the_link"
        # 创建 WebDriver 实例
        driver = webdriver.Chrome()
        driver.get(url)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "lxml")

        # 获取页面中全部的图片及其对应的描述；
        all_elements = soup.find_all("div", class_="pic")
        all_img_urls = [i.find_all("img")[0].get("src") for i in all_elements]
        all_img_captions = [i.find_all("a")[-1].text for i in all_elements]

        pathImg = "textiles/img/" + categ_id_name[categ] + "/"
        if not os.path.exists(pathImg):
            os.makedirs(pathImg)

        pathCap = "textiles/cap/" + categ_id_name[categ] + "/"
        if not os.path.exists(pathCap):
            os.makedirs(pathCap)

        # 写入图片、文本至本地；
        for idx, i in enumerate(all_img_urls):
            try:
                urllib.request.urlretrieve(i, pathImg + str(page) + "-" + str(idx) + ".png")
                with open(pathCap + str(page) + "-" + str(idx) + ".txt", "w", encoding="utf8") as f:
                    f.write(all_img_captions[idx])
                time.sleep(0.1)
                # print(1)
            except:
                print(f"第{page}页未写入...")
                continue
        print("已写入图片...")
        print("已写入描述...")

        time.sleep(1)