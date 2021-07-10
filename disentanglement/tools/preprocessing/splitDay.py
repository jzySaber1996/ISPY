import datetime
"""
将一个长聊天日志文件转换为按天存储的日志文件
"""
"""
@:param str:string
@:return strptime:datatime
"""
def strToDatetime(str):
    strptime = datetime.datetime.strptime(str, "%Y-%m-%d")
    return strptime
path = "/home/yuminz/gitter_chatmessage/scala_scala/"
fileName = "2016-01-01"
file = open(path+fileName+".txt","w",encoding="UTF-8")
for line in open("/home/yuminz/gitter_chatmessage/scala_scala_chatFormat.ascii.txt",encoding="UTF-8"):
    day = line[1:11]
    date = strToDatetime(day)
    fileDate = strToDatetime(fileName)
    if fileDate != date:
        file = open(path+day+".txt","w",encoding="UTF-8")
        fileName = day
        file.write(line)
    else:
        file.write(line)





