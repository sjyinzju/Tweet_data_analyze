import json
import csv
import os
import glob

def extract_detailed_tweet_data(tweet_obj):
    data = {
        'tweet_id': '',
        'tweet_content': '',
        'tweet_posted_time': '',
        'tweet_language': '',
        'favorites_count': 0,
        'retweet_count': 0,
        'user_name': '',
        'user_followers_count': 0,
        'user_statuses_count': 0,
        'location_name': '',
        'latitude': '',
        'longitude': '',
        'in_reply_to_status_id': '',
        'in_reply_to_user_id': '',
        'in_reply_to_screen_name': ''
    }
    try:
        data['tweet_id'] = tweet_obj.get('id', '')
        
        data['tweet_content'] = tweet_obj.get('body', tweet_obj.get('text', ''))
        
        data['tweet_posted_time'] = tweet_obj.get('postedTime', tweet_obj.get('created_at', ''))
        
        data['tweet_language'] = tweet_obj.get('twitter_lang', tweet_obj.get('lang', ''))
        
        data['favorites_count'] = tweet_obj.get('favoritesCount', 0)
        data['retweet_count'] = tweet_obj.get('retweetCount', 0)
        
        if data['favorites_count'] == 0 and data['retweet_count'] == 0:
            if 'actor' in tweet_obj and isinstance(tweet_obj['actor'], dict):
                actor = tweet_obj['actor']
                user_favorites = actor.get('favoritesCount', 0)
                if data['favorites_count'] == 0 and user_favorites > 0:
                    data['favorites_count'] = user_favorites        
        if 'actor' in tweet_obj and isinstance(tweet_obj['actor'], dict):
            actor = tweet_obj['actor']
            data['user_name'] = actor.get('preferredUsername', actor.get('screen_name', ''))
            data['user_followers_count'] = actor.get('followersCount', 0)
            data['user_statuses_count'] = actor.get('statusesCount', 0)
        
        if 'location' in tweet_obj and isinstance(tweet_obj['location'], dict):
            location = tweet_obj['location']
            data['location_name'] = location.get('displayName', '')
            
            if 'geo' in location and isinstance(location['geo'], dict):
                geo = location['geo']
                if 'coordinates' in geo:
                    coords = geo['coordinates']
                    if isinstance(coords, list):
                        if geo.get('type') == 'Point' and len(coords) >= 2:
                            data['latitude'] = coords[0]
                            data['longitude'] = coords[1]
                        elif geo.get('type') == 'Polygon' and len(coords) > 0 and len(coords[0]) >= 2:
                            data['latitude'] = coords[0][0][1]
                            data['longitude'] = coords[0][0][0]
        
        if not data['latitude'] and not data['longitude']:
            if 'geo' in tweet_obj and isinstance(tweet_obj['geo'], dict):
                geo = tweet_obj['geo']
                if 'coordinates' in geo:
                    coords = geo['coordinates']
                    if isinstance(coords, list) and len(coords) >= 2:
                        if geo.get('type') == 'Point':
                            data['latitude'] = coords[0]
                            data['longitude'] = coords[1]
        
        if 'inReplyTo' in tweet_obj and isinstance(tweet_obj['inReplyTo'], dict):
            in_reply_to = tweet_obj['inReplyTo']
            if 'link' in in_reply_to:
                link = in_reply_to['link']
                if isinstance(link, str):
                    import re
                    match = re.search(r'twitter\.com/([^/]+)/statuses/(\d+)', link)
                    if match:
                        data['in_reply_to_screen_name'] = match.group(1)
                        data['in_reply_to_status_id'] = match.group(2)
            if 'id' in in_reply_to:
                data['in_reply_to_user_id'] = in_reply_to['id']
        
    except Exception as e:
        pass

    return data
def convert_detailed_twitter_json_to_csv(json_folder_path, csv_file_path):
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    
    if not json_files:
        return
    
    
    fieldnames = [
        'tweet_id', 'tweet_content', 'tweet_posted_time', 'tweet_language',
        'favorites_count', 'retweet_count', 'user_name', 'user_followers_count',
        'user_statuses_count', 'location_name', 'latitude', 'longitude',
        'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name'
    ]
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_records = 0
        
        for i, json_file in enumerate(json_files):
            if i % 100 == 0:
                pass

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            obj = json.loads(line)
                            
                            records = []
                            if isinstance(obj, list):
                                records.extend(obj)
                            elif isinstance(obj, dict):
                                if 'body' in obj or 'text' in obj:
                                    records.append(obj)
                                elif 'tweets' in obj and isinstance(obj['tweets'], list):
                                    records.extend(obj['tweets'])
                                else:
                                    records.append(obj)
                            
                            for record in records:
                                extracted_data = extract_detailed_tweet_data(record)
                                writer.writerow(extracted_data)
                                total_records += 1
                                
                        except json.JSONDecodeError:
                            continue
                                    
            except Exception as e:
                continue
    

if __name__ == "__main__":
    json_folder = r"C:\workplace\tweets_2"
    output_csv = r"C:\workplace\detailed_tweets_with_replies.csv"    
    convert_detailed_twitter_json_to_csv(json_folder, output_csv)