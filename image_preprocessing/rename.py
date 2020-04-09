import os

directory = "E:/Docs/Downloads/datasets/test_annotations"

i=0
for filename in os.listdir(directory):
    if filename.endswith(".bmp"):
         src = os.path.join(directory, filename)
         dst = os.path.join(directory, str(i)+'.bmp')
         print(src)
         print(dst)
         i=i+1
         os.rename(src, dst)
         
         #gray_anotate(path)
         #decode_gray(path)
         
        #continue
    else:
        continue
