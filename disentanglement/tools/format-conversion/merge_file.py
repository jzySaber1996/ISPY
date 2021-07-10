"""
将以天为单位存储的解耦后文件合并成一个文件
"""
import os
path = "/home/yuminz/message/scala_scala/"
dir = os.listdir(path)
file = open("/home/yuminz/message/scala_scala.txt","w",encoding="UTF-8")
dir.sort()
for fileName in dir:
    row = 1
    for line in open(path+fileName):
        if line.startswith("------") :
            file.write(line)
            row =1
            print(line)
        else:
          line =str(row)+" "+line
          row = row + 1
          file.write(line)
          print(line)
file.close()
