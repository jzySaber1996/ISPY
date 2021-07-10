with open("/home/yuminz/annotations/gitlab.ascii.txt",'w',encoding="utf-8") as f:
    for line in open("/home/yuminz/annotations/example-run-gitlabbak_message.3.out",encoding="UTF-8"):
        splits = line.split(":")
        f.write(splits[1])