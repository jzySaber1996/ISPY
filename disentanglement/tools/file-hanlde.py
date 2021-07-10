import json
import os
import re
if __name__ == '__main__':
    path="/home/yuminz/FreeNode"
    filedir = os.listdir(path)
    for file in filedir:
        with open("/home/yuminz/FreenodeNew/"+file,'w',encoding="UTF-8") as f:
           print(file)
           for line in open(path+"/"+file,encoding="UTF-8"):
            # if re.search('\<[\s\S]*\>',line):
            #     try:
            #         lines = line.split(" <")
            #         line = "["+lines[0]+"] <"+lines[1]
            #         f.write(line)
            #     except :
            #          print(line)

             newline = line.strip('\n')
             d = json.loads(newline)
             user =d["_values"]["user"]
             timestamp = d["_values"]["timstamp"]
             body = d["_values"]["body"]
             body = body.strip()
             user = user.strip()
             if body != "" and user!="":
                 times = timestamp.split("T")
                 inputLine="["+timestamp+"] <"+user+"> "+body+'\n'
                 f.write(inputLine)



