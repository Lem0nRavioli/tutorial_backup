import pytube

# this url is the tutorial at the origin of this script
url = "https://www.youtube.com/watch?v=Jelbqbs80Ms&ab_channel=Mr_MoMoMusic"

video = pytube.YouTube(url)
print(video.streams.filter(type="audio"))
stream = video.streams.filter(type='audio').order_by('abr').desc()
print(stream)
# for stream in video.streams:
#     if "video" in str(stream) and "mp4" in str(stream):
#         print(stream)
top = video.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
print(top.resolution)

# stream = video.streams.get_by_itag(22)  # corresponding to 720p
# stream.download(filename="video_dl_example")


# https://youtu.be/UM6YDJ2aalU?t=426



'''# Downloading playlist

url_pl = "https://youtube.com/playlist?list=PLbfP7G2t2X4DEBhkl3afQhPODAvVMfUjI"
playlist = pytube.Playlist(url_pl)
print(playlist)
# playlist.download_all()  # not working
for url in playlist:
    try:
        print(url)
        video = pytube.YouTube(url)
        stream = video.streams.get_by_itag(22)
        stream.download()
    except:
        print(f"Issue downloading the video {url}")
        continue
'''
