"""
为ｇitter聊天纪录里面的时间加上Ｔ
"""
file = open("/home/yuminz/gitter_chatmessage/Dogfalo_materialize_chatFormat.txt","w",encoding="UTF-8")
for line in open("/home/yuminz/gitter_chatmessage/Dogfalo_materialize_chatFormat.raw.txt"):
    splits = line.split(" ",1)
    file.write(splits[0]+"T"+splits[1])

file.close()