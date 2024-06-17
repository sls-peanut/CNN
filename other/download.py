# -*- coding = utf-8 -*-
# @Time :2024/6/17 17:13
# @Author:sls
# @FIle:download.py
# @Annotation:
import requests
import os
import time
import requests
def download_file(url, file_name):
    """
    下载指定 URL 的文件并保存到本地
    """
    # 获取文件大小
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))

    # 创建文件夹
    os.makedirs('downloads', exist_ok=True)
    file_path = os.path.join('downloads', file_name)

    # 下载文件
    downloaded = 0
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1MB 的块大小
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = round(downloaded / file_size * 100, 2)
                    speed = downloaded / (time.time() - start_time) / 1024 / 1024
                    print(f'已下载 {downloaded}/{file_size} 字节 ({percent}%), 下载速度: {speed:.2f} MB/s', end='\r')

    print(f'\n文件 "{file_name}" 下载完成!')
    return file_path

url = 'https://rdm.inesctec.pt/dataset/604dfdfa-1d37-41c6-8db1-e82683b8335a/resource/df04ea95-36a7-49a8-9b70-605798460c35/download/breasthistology.zip'
file_name = 'breasthistology.zip'
download_file(url, file_name)