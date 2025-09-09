import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
import song_name_extractor
import time
import json
from datetime import datetime
from utils.config import getAPIValue


class YouTubePianoCoverDataset:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.dataset = []

    def search_piano_covers(self, query, max_results=50):
        """
        搜索钢琴改编曲
        """
        try:
            search_response = self.youtube.search().list(
                q=query,
                part='snippet',
                maxResults=max_results,
                type='video',
                videoDuration='medium',  # 中等长度视频 (4-20分钟)
                order='viewCount'  # 按观看量排序
            ).execute()

            return search_response.get('items', [])

        except HttpError as e:
            print(f'An HTTP error occurred: {e}')
            return []

    def get_video_details(self, video_id):
        """
        获取视频详细信息
        """
        try:
            video_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()

            return video_response.get('items', [])[0] if video_response.get('items') else None

        except HttpError as e:
            print(f'An HTTP error occurred: {e}')
            return None

    def get_video_url(self, video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    def build_dataset(self, queries, max_results_per_query=20):
        """
        构建数据集
        """
        for query in queries:
            print(f"querying: {query}")
            videos = self.search_piano_covers(query, max_results_per_query)

            titles = list(map(lambda a: a['snippet']['title'], videos))
            names = song_name_extractor.extract_name(titles)
            print(names)

            for video, name in zip(videos, names):
                video_id = video['id']['videoId']
                print(f"process video: {video['snippet']['title']} (ID: {video_id})")

                # 获取视频详情
                video_details = self.get_video_details(video_id)
                if not video_details:
                    continue

                # 生成URL
                video_url = self.get_video_url(video_id)

                # 收集数据
                data_entry = {
                    'cover_id': video_id,
                    'cover_title': video['snippet']['title'],
                    'cover_channel': video['snippet']['channelTitle'],
                    'cover_publish_date': video['snippet']['publishedAt'],
                    'cover_description': video['snippet']['description'],
                    'cover_duration': video_details['contentDetails']['duration'],
                    'cover_view_count': video_details['statistics'].get('viewCount', 0),
                    'cover_like_count': video_details['statistics'].get('likeCount', 0),
                    'cover_comment_count': video_details['statistics'].get('commentCount', 0),
                    'original_song': name,
                    'search_query': query,
                    'collected_date': datetime.now().isoformat(),
                    'video_url': video_url,
                }

                self.dataset.append(data_entry)

                # 避免API限制
                time.sleep(0.1)

        return self.dataset

    def save_to_csv(self, filename='piano_covers_dataset.csv'):
        """
        保存数据集到CSV文件
        """
        if not self.dataset:
            print("没有数据可保存")
            return

        df = pd.DataFrame(self.dataset)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"数据集已保存到 {filename}")

    def save_to_json(self, filename='piano_covers_dataset.json'):
        """
        保存数据集到JSON文件
        """
        if not self.dataset:
            print("没有数据可保存")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"数据集已保存到 {filename}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的YouTube API密钥
    API_KEY = getAPIValue('youtube')

    # 搜索查询列表
    search_queries = [
        "piano cover",
        "piano arrangement",
        "piano version",
    ]

    # 创建数据集构建器
    dataset_builder = YouTubePianoCoverDataset(API_KEY)

    # 构建数据集
    dataset = dataset_builder.build_dataset(search_queries, max_results_per_query=2)

    # 保存数据集
    #dataset_builder.save_to_csv()
    #dataset_builder.save_to_json()

    print(f"共收集 {len(dataset)} 个钢琴改编曲视频")