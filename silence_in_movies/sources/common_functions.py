import os
from pymediainfo import MediaInfo




def find_downloaded_movies(raw_path):

    movies_folders = [f for f in os.listdir(raw_path) if not f.startswith('.')]
    # Find all downloaded movies
    downloaded = []

    for m in movies_folders:
        movie_folder = raw_path/m
        if len([f for f in os.listdir(movie_folder) if not f.startswith('.')]) == 1:
            downloaded.append(movie_folder)

    return downloaded


def get_movie_file_dict(raw_path):
        # Get mp4/mkv files
    movie_files = {}

    downloaded_movies = find_downloaded_movies(raw_path)            
    for m in downloaded_movies:
        movie_id = str(m).split('/')[-1]
        ref_folder = [f for f in os.listdir(m) if not f.startswith('.')]
        torrent_folder = m/(ref_folder[0])
        movie_file = [f for f in os.listdir(torrent_folder) if f.endswith('.mkv') 
                        or f.endswith('.mp4') or f.endswith('.m4v') or f.endswith('.avi')][0]
        movie_files[movie_id] = torrent_folder/movie_file
        
    return movie_files


def get_movies_status(raw_path, prep_path):

    downloaded_movies = find_downloaded_movies(raw_path)
    movie_file_dict = get_movie_file_dict(raw_path)
    
    movie_status = {}

    # Check if movies have all files needed
    for d in downloaded_movies:
        movie_id = str(d).split('/')[-1]
        movie_status[movie_id] = []
        prep_folder = prep_path/movie_id

        file_path = str(movie_file_dict[movie_id])

        media_info = MediaInfo.parse(file_path)
        #duration in milliseconds
        movie_duration = media_info.tracks[0].duration//1000 
        
        if not os.path.exists(prep_folder/'torrent_info.json'):
            movie_status[movie_id].append('json')

        if not os.path.exists(prep_folder/'audio/movie_audio.mp3'):
            movie_status[movie_id].append('audio')

        if not os.path.exists(prep_folder/'subtitle'):
            movie_status[movie_id].append('subtitle')
        elif len(os.listdir(prep_folder/'subtitle/')) == 0:
                movie_status[movie_id].append('subtitle')
        
        if not os.path.exists(prep_folder/'images/'):
            movie_status[movie_id].append('images')
        else:
            expected_frames = movie_duration//10
            found_frames = len(os.listdir(prep_folder/'images/'))
            if abs(expected_frames - found_frames) > 10:
                print(expected_frames)
                print(found_frames)
                movie_status[movie_id].append('images')

    return movie_status


