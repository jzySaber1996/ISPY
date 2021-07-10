"""讲解耦后输出的ｏut文件转为以天为单位的ａnnotation文件"""

path = "/home/yuminz/annotations/scala_scala/"
file = open(path+"2016-01-01.annotation.txt","w",encoding="UTF-8")
filetemp="2016-01-01.annotation.txt"
for line in open("/home/yuminz/PythonProject/disentanglement/src/scala_scala.1.out",encoding="UTF-8"):
    if line.startswith("#"):
            continue
    splits = line.split(":")
    fileName = splits[0].split("/")[-1]
    anno = splits[-1]
    if filetemp != fileName:
        file.close()
        file = open(path+fileName,"w",encoding="UTF-8")
        filetemp = fileName
        file.write(anno)
    else:
        file.write(anno)
file.close()
