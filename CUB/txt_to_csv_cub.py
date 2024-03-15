import csv

TXT_DIR = '/home/haen/workspace/Imagen_org/cub-caption/'

res = []
for i in range(11788):
    f = open(TXT_DIR +str(i) + ".txt", 'r')
    line = f.readline()
    res.append([str(i)+'.jpg',line])
    f.close()

with open('cub_caption.csv','w') as file :
    write = csv.writer(file)
    write.writerows(res)