import urllib.request
import zipfile

# 下载zip文件
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # 替换为你的文件URL
file_name = "tiny-imagenet-200.zip"  # 保存的文件名
urllib.request.urlretrieve(url, file_name)

# 解压zip文件
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall('../tiny-imagenet-200')  # 解压到指定文件夹
