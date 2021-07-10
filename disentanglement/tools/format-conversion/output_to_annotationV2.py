path = "/home/yuminz/gitter_chatmessage/"
file = open(path+"Dogfalo_materialize_chatFormat.annotation.txt","w",encoding="UTF-8")
for line in open("../../src/Dogfalo_materizlize.1.out"):
    if line.startswith("#"):
        continue
    splits = line.split(":")
    file.write(splits[1])
file.close()