import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
import time
import json
from datetime import datetime

prompt = "你是一个音乐信息提取助手，请从以下文本中精确提取歌曲名称和艺人名称，只返回JSON格式的结果："


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

    def extract_original_song_info(self, title, description):
        """
        尝试从标题和描述中提取原曲信息
        这是一个简单的实现，实际应用中可能需要更复杂的NLP技术
        """
        # 常见标识符
        indicators = ['cover', 'piano cover', 'by', 'original', 'from']

        # 简单尝试移除常见钢琴改编标识
        original_title = title.lower()
        for indicator in indicators:
            original_title = original_title.replace(indicator, '')

        # 移除括号内容（通常包含cover等信息）
        import re
        original_title = re.sub(r'\([^)]*\)', '', original_title)
        original_title = re.sub(r'\[[^\]]*\]', '', original_title)

        # 清理多余空格
        original_title = ' '.join(original_title.split())

        return original_title.strip()

    def build_dataset(self, queries, max_results_per_query=20):
        """
        构建数据集
        """
        for query in queries:
            print(f"搜索查询: {query}")
            videos = self.search_piano_covers(query, max_results_per_query)

            for video in videos:
                video_id = video['id']['videoId']
                print(f"处理视频: {video['snippet']['title']} (ID: {video_id})")

                # 获取视频详情
                video_details = self.get_video_details(video_id)
                if not video_details:
                    continue

                # 提取原曲信息
                original_song = self.extract_original_song_info(
                    video['snippet']['title'],
                    video['snippet']['description']
                )

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
                    'original_song_title': original_song,
                    'search_query': query,
                    'collected_date': datetime.now().isoformat()
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
    API_KEY = "AIzaSyCaqiqv2UFG0N9Y_C1U-jPrzRmDg1Vtuug"

    # 搜索查询列表
    search_queries = [
        "piano cover",
        "piano arrangement",
        "piano version",
        "piano instrumental"
    ]

    # 创建数据集构建器
    dataset_builder = YouTubePianoCoverDataset(API_KEY)

    # 构建数据集
    dataset = dataset_builder.build_dataset(search_queries, max_results_per_query=20)

    # 保存数据集
    dataset_builder.save_to_csv()
    dataset_builder.save_to_json()

    print(f"共收集 {len(dataset)} 个钢琴改编曲视频")