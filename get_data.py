import os

def download(url):
    os.system(f"youtube-dl -x --audio-format wav\
    --postprocessor-args \"-ac 1 -ar 16000 -t 00:10:00\" {url}")

cello = "https://www.youtube.com/watch?v=Ivbed1nybRU"
trombone = "https://www.youtube.com/watch?v=IrezeozOhUY",\
           "https://www.youtube.com/watch?v=J_U14MSWjFM"
flute = "https://www.youtube.com/watch?v=bfoQzlMNDNU"

if __name__ == '__main__':
    download(cello)
    for i,elm in enumerate(trombone):
        download(elm)
    download(flute)
