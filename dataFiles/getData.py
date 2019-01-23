import io
import pandas as pd

video_data = pd.read_csv("MSRVideoDescription.csv", sep=',', encoding='utf8')
print(video_data.columns)
print(video_data["Description"])
print(video_data.iloc[71631])
video_data['Description'] = video_data.apply(lambda row: row['Description'].replace('\n', ''), axis=1)
video_data['Description'] = video_data.apply(lambda row: row['Description'].replace('\r', ''), axis=1)

video_data.to_csv("MSRVideoDescription2.csv", index=False)

f1 = io.open("MSRVideoDescription2.csv", "r", encoding="utf-8")
#  cont = f1.read().split("\n")
cont = f1.readlines()
#  print(cont)

f2 = open("youtube_mapping.txt", "r")
maps = {}
ans = f2.readlines()
for l in ans:
    val1, val2 = l.strip().split()
    maps[val1] = val2
    

ans = open("MSRVideoDescriptionNew.csv", "w")
for l in cont:
    l = l.strip()
    st = l.split(",")
    print(st)
    print(l)
    if len(st) > 1:
        #  print(len(st))
        #  print(st)
        assert(len(st) >= 8)
        if "English" in l:
            vidId = st[0] + "_" + st[1] + "_" + st[2]
            bool1  = ("unverified" in l)
            bool3 = vidId in maps
            bool4 = bool3 and int(maps[vidId][3:]) < 1201
            bool5 = bool3 and int(maps[vidId][3:]) >= 1201
            if (bool4) and (not bool1):
                ans.write(l)
                ans.write("\n")
            elif bool5:
                ans.write(l)
                ans.write("\n")

ans.close()



