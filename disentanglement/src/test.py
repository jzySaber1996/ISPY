X="2004-11-15_03.annotation.txt 2005-06-27_12.annotation.txt 2005-08-08_01.annotation.txt 2008-12-11_11.annotation.txt 2009-02-23_10.annotation.txt 2009-03-03_10.annotation.txt 2009-10-01_17.annotation.txt 2011-05-29_19.annotation.txt 2011-11-13_02.annotation.txt 2016-12-19_20.annotation.txt"
str_list = X.split(" ")

for s in str_list:
    print('../data/dev/'+s,end=' ')

print('\n')
print(str_list)
print(len(str_list))

