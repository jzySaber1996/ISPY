#!/bin/bash

＃　　批处理　映射annotaion文件为实际聊天对话
for file in `ls /home/yuminz/gitter_chatmessage/scala_scala/ | grep ascii.txt`
do 
    python3 graph-to-messages.py  /home/yuminz/gitter_chatmessage/scala_scala/$file  /home/yuminz/annotations/scala_scala/$file
    echo $file
done

