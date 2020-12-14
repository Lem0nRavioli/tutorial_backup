import pytube

# this url is the tutorial at the origin of this script
url = "https://www.youtube.com/watch?v=UM6YDJ2aalU&ab_channel=NeuralNine"

video = pytube.YouTube(url)

for stream in video.streams:
    if "video" in str(stream) and "mp4" in str(stream):
        print(stream)

stream = video.streams.get_by_itag(22)  # corresponding to 720p
stream.download(filename="video_dl_example")


# https://youtu.be/UM6YDJ2aalU?t=426