import ast
import json
import re
import time
from datetime import datetime
from fuzzywuzzy import fuzz, process

import pandas as pd
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils.config import getAPIValue


class YouTubePianoCoverDataset:
    def __init__(self, youtube_api, google_llm_api):
        self.youtube = build('youtube', 'v3', developerKey=youtube_api)
        self.client = genai.Client(api_key=google_llm_api)
        self.dataset = []

    def search_piano_covers(self, query, max_results=50, result_per_search=50):
        target_len = max_results
        next_token = None
        result_set = []
        while target_len > 0:
            result = min(result_per_search, target_len)

            try:
                search_response = self.youtube.search().list(
                    q=query,
                    pageToken=next_token,
                    part='snippet',
                    maxResults=result,
                    type='video',
                    videoDuration='medium',  # 中等长度视频 (4-20分钟)
                ).execute()

                result_set.extend(search_response.get('items', []))
                next_token = search_response.get('nextPageToken')

                target_len -= result_per_search
                time.sleep(0.1)
            except HttpError as e:
                print(f'An HTTP error occurred: {e}')
                return result_set
        return result_set

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

    def build_dataset(self, queries, max_results_per_query=20, result_per_search=50):
        """
        构建数据集
        """
        for query in queries:
            print(f"querying: {query}")
            videos = self.search_piano_covers(query, max_results_per_query, result_per_search)

            for video in videos:
                video_id = video['id']['videoId']
                print(f"process video: {video['snippet']['title']} (ID: {video_id})")

                # enable video_details: not recommended due to high api cost
                # video_details = self.get_video_details(video_id)
                # if not video_details:
                # continue
                # time.sleep(0.1)

                # 生成URL
                video_url = self.get_video_url(video_id)

                # 收集数据
                data_entry = {
                    'cover_id': video_id,
                    'cover_title': video['snippet']['title'],
                    'cover_channel': video['snippet']['channelTitle'],
                    'cover_publish_date': video['snippet']['publishedAt'],
                    'cover_description': video['snippet']['description'],
                    # 'cover_duration': video_details['contentDetails']['duration'],
                    # 'cover_view_count': video_details['statistics'].get('viewCount', 0),
                    # 'cover_like_count': video_details['statistics'].get('likeCount', 0),
                    # 'cover_comment_count': video_details['statistics'].get('commentCount', 0),
                    'search_query': query,
                    'collected_date': datetime.now().isoformat(),
                    'video_url': video_url,
                }

                self.dataset.append(data_entry)

        return self.dataset

    def extract_name(self, titles, time_delay=0.1):
        contents = f'you are a music collection assistant, please extract the EXACT music name without adding any ' \
                   f'authors\' name from the following video titles, considering carefully about the music name words and ' \
                   f'formats. return in List format:{titles}'
        # print(contents)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
            ),
        )
        cleaned_text = re.sub(r'^[^\[]*|[^\]]*$', '', response.text)
        print(cleaned_text)
        time.sleep(time_delay)
        return ast.literal_eval(cleaned_text)

    def extract_original_title(self, csv_file_path, overwrite=False, batch_size=50, similarity_threshold=0.7):
        df = pd.read_csv(csv_file_path)
        start_idx = 0
        names = []

        if not overwrite and 'original_song' in df.columns:
            non_empty_mask = df['original_song'].notna() & (df['original_song'] != '')
            if non_empty_mask.any():
                start_idx = non_empty_mask.sum()
                names = list(df['original_song'])[0:start_idx]
                print(f"starting from {start_idx}")

        titles = df['cover_title']
        if len(titles) == start_idx: return
        titles_split = [titles[i:i + batch_size] for i in range(start_idx, len(titles), batch_size)]

        try:
            for batch_idx, batch_titles in enumerate(titles_split):
                # 提取当前批次的名称
                extracted_names = self.extract_name(batch_titles)

                # 处理每个提取的名称
                batch_results = []
                retire_list = []
                for i, (original_title, extracted_name) in enumerate(zip(batch_titles, extracted_names)):
                    # check availability
                    if not extracted_name or pd.isna(extracted_name):
                        batch_results.append(pd.NA)
                        continue

                    # add to matching list
                    retire_list.append(extracted_name)

                    failed = False
                    # using edit distance, check all failed name before corresponding pair,
                    # remove matched name and everything before it
                    for n in retire_list:
                        similarity = fuzz.partial_ratio(n.lower(), original_title.lower())

                        # test similarity
                        if (similarity / 100) >= similarity_threshold:
                            failed = False
                            batch_results.append(extracted_name)
                            # clean everything before the success matching
                            retire_list = retire_list[retire_list.index(n) + 1:]
                            print(f"matched: '{n}' -> '{original_title}' (similarity: {similarity}%)")
                            continue
                        else:
                            failed = True
                            print(f"failed: '{n}' -> '{original_title}' (similarity: {similarity}%)")

                    if failed:
                        batch_results.append(pd.NA)

                names.extend(batch_results)
        except Exception as e:
            print(f"error when communicating with LLM {e} \n saving solved work...")
            pass
        print(names)

        name_df = pd.DataFrame(names, columns=['original_song'])
        combined_df = pd.concat([df, name_df], axis=1)

        # save
        output_filename = f"piano_covers_with_original_titles.csv"
        combined_df.to_csv(output_filename, index=False, encoding='utf-8')

        print(f"new data set have been saved to {output_filename}")
        return combined_df

    def find_original_songs(self, csv_file_path, max_results_per_song=1, top_n=-1):
        """
        根据CSV文件中的original_song_title字段搜索原曲视频
        """
        k = top_n
        df = pd.read_csv(csv_file_path)

        original_songs_data = []

        for index, row in df.iterrows():
            original_song_title = row['original_song']

            # 跳过空值或无效的歌曲标题
            if pd.isna(original_song_title) or not original_song_title.strip():
                print(f"跳过第 {index + 1} 行: 无有效原曲标题")
                original_songs_data.append({})
                continue

            print(f"搜索原曲: {original_song_title} (第 {index + 1}/{len(df)} 行)")

            # 搜索原曲视频
            try:
                # 添加"official"等关键词提高找到原曲的概率
                search_query = f"{original_song_title} official"
                search_results = self.search_piano_covers(search_query, max_results=max_results_per_song)

                if not search_results:
                    print(f"未找到原曲: {original_song_title}")
                    original_songs_data.append({})
                    continue

                # 获取第一个结果的详细信息
                original_video_id = search_results[0]['id']['videoId']
                # original_video_details = self.get_video_details(original_video_id)

                # if not original_video_details:
                # print(f"无法获取原曲详情: {original_song_title}")
                # original_songs_data.append({})
                # continue

                # 提取原曲信息
                original_data = {
                    'original_video_id': original_video_id,
                    'original_title': search_results[0]['snippet']['title'],
                    'original_publish_date': search_results[0]['snippet']['publishedAt'],
                    'original_description': search_results[0]['snippet']['description'],
                    # 'original_duration': original_video_details['contentDetails']['duration'],
                    # 'original_comment_count': original_video_details['statistics'].get('commentCount', 0),
                    'original_video_url': self.get_video_url(original_video_id),
                    'original_search_query': search_query
                }

                original_songs_data.append(original_data)
                print(f"find origin: {original_data['original_title']}")

            except Exception as e:
                print(f"error when finding origin: {e}")
                original_songs_data.append({})

            # avoid api restriction
            time.sleep(0.1)
            if k > 0:
                k -= 1
            elif k == 0:
                break

        # combine to DataFrame
        original_df = pd.DataFrame(original_songs_data)
        combined_df = pd.concat([df, original_df], axis=1)

        # save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"piano_covers_with_originals_{timestamp}.csv"
        combined_df.to_csv(output_filename, index=False, encoding='utf-8')

        print(f"new data set have been saved to {output_filename}")
        return combined_df

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
    API_KEY = getAPIValue('youtube')

    # queries list
    search_queries = [
        "piano cover",
        "piano arrangement",
        "piano version",
    ]

    # create constructor
    dataset_builder = YouTubePianoCoverDataset(API_KEY)

    # build dataset
    dataset = dataset_builder.build_dataset(search_queries, max_results_per_query=200)

    # save to csv
    dataset_builder.save_to_csv()
    dataset_builder.save_to_json()

    print(f"collect {len(dataset)} piano covers")
