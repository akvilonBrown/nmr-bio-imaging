import os
#import glob

#directory = "E:/Docs/Downloads/datasets/junk/label"
directory = "E:/Projects/NMR/grandset/label"


'''
txtfiles = []
for file in glob.glob("*.bmp"):
    txtfiles.append(file)
'''  
'''
filelist = os.listdir(directory)
print(type(filelist))
print(filelist)
'''

#directory = "E:/Projects/NMR/drive-download-20200325T173948Z-001/train_painted"

#i=0
for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".bmp"):
         src = os.path.join(directory, filename)
         dst = os.path.join(directory, 'C_batch_'+ str(i).zfill(5)+'.bmp')
         print(src)
         print(dst)
         #i=i+1
         os.rename(src, dst)
         
         #gray_anotate(path)
         #decode_gray(path)
         
        #continue
    else:
        continue
