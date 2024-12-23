import csv
import requests
import time
import os

current_dir = os.getcwd()
print(current_dir)
new_dir = os.path.dirname(os.path.abspath(__file__))
print(new_dir)
os.chdir(new_dir)

# 文件路径
input_path = r'rows_2001_to_2500.csv'
output_path = r'following.csv'
processed_ids_path = r'processed_ids.txt'

# GitHub API Token（请将其替换为你自己的Token）
GITHUB_TOKEN = '***********'

# 设置认证头
headers = {
    'Authorization': f'token {GITHUB_TOKEN}'
}

# 从CSV文件读取用户数据
actor_ids = []
with open(input_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        actor_ids.append((int(row[0]), row[1]))  # 第一列为开发者 ID，第二列为用户名

# 加载已处理的用户ID
processed_ids = set()
try:
    with open(processed_ids_path, 'r', encoding='utf-8') as f:
        processed_ids = set(int(line.strip()) for line in f)
except FileNotFoundError:
    print("No processed IDs file found. Starting fresh.")

# GitHub API基本URL
base_url = "https://api.github.com/users/{}/following"

# 获取某个GitHub用户的关注列表
def get_following(user_id):
    url = f"https://api.github.com/user/{user_id}/following"
    following = []
    page = 1

    while True:
        response = requests.get(url, headers=headers, params={'page': page, 'per_page': 100})
        if response.status_code == 200:
            data = response.json()
            if not data:  # 如果没有数据，说明所有关注数据已获取
                break
            following.extend([(user['id'], user['login']) for user in data])
            page += 1
        elif response.status_code == 404:
            print(f"Error fetching following for user ID {user_id}: 404 - Not Found")
            break
        else:
            print(f"Error fetching following for user ID {user_id}: {response.status_code}")
            break
        time.sleep(120)  # 避免请求过于频繁
    return following

# 写入关注数据到CSV文件，并记录已处理的用户ID
with open(output_path, 'a', newline='', encoding='utf-8') as csvfile, open(processed_ids_path, 'a', encoding='utf-8') as idfile:
    writer = csv.writer(csvfile)
    
    # 如果文件是空的，写入表头
    if csvfile.tell() == 0:
        writer.writerow(['user_id', 'user_name', 'following_user_id', 'following_user_name'])

    for user_id, user_name in actor_ids:
        if user_id in processed_ids:
            print(f"Skipping {user_name} ({user_id}), already processed.")
            continue

        print(f"Fetching following data for {user_name} ({user_id})...")
        following_data = get_following(user_id)
        
        for following_user_id, following_user_name in following_data:
            writer.writerow([user_id, user_name, following_user_id, following_user_name])
        
        # 将已处理的用户 ID 写入文件并同步到内存
        idfile.write(f"{user_id}\n")
        idfile.flush()
        processed_ids.add(user_id)

        # 控制请求速率
        time.sleep(60)

print("Data fetching and writing completed.")
