import numpy as np
import skimage
from PIL import Image

img = Image.open('balloons.jpg').convert('L')
img.save('gray.jpg')

im = Image.open("gray.jpg")
h = im.size[0]
w = im.size[1]
r = 128
im = im.resize([r,r])
pixels = im.load()
x = np.asarray(im)
n = int(r/64) 
p = int(r/8)

img_blocks = skimage.util.view_as_blocks(x, block_shape=(8, 8))

H1 = [[0.5,0,0,0,0.5,0,0,0],
      [0.5,0,0,0,-0.5,0,0,0],
      [0,0.5,0,0,0,0.5,0,0],
      [0,0.5,0,0,0,-0.5,0,0],
      [0,0,0.5,0,0,0,0.5,0],
      [0,0,0.5,0,0,0,-0.5,0],
      [0,0,0,0.5,0,0,0,0.5],
      [0,0,0,0.5,0,0,0,-0.5]
      ]

H2 = [
      [0.5,0,0.5,0,0,0,0,0],
      [0.5,0,-0.5,0,0,0,0,0],
      [0,0.5,0,0.5,0,0,0,0],
      [0,0.5,0,-0.5,0,0,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1]
      ]

H3 = [[0.5,0.5,0,0,0,0,0,0],
      [0.5,-0.5,0,0,0,0,0,0],
      [0,0,1,0,0,0,0,0],
      [0,0,0,1,0,0,0,0],
      [0,0,0,0,1,0,0,0],
      [0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,1]
      ]

for i in range(len(H1)):
    for j in range(len(H1[i])):
        if(H1[i][j]==0.5):
            H1[i][j] = 0.7
        elif(H1[i][j]==-0.5):
            H1[i][j] = -0.7

for i in range(len(H1)):
    for j in range(len(H1[i])):
        if(H2[i][j]==0.5):
            H2[i][j] = 0.7
        elif(H2[i][j]==-0.5):
            H2[i][j] = -0.7
            
for i in range(len(H3)):
    for j in range(len(H3[i])):
        if(H3[i][j]==0.5):
            H3[i][j] = 0.7
        elif(H3[i][j]==-0.5):
            H3[i][j] = -0.7
        
images = []
H = np.dot(np.dot(H1,H2),H3)

new_img_blocks = [[[] for j in range(p)] for i in range(p)]
for i in range(len(img_blocks)): 
    for j in range(len(img_blocks)):
        new_img_blocks[i][j] = np.dot(np.dot(H.T,img_blocks[i][j]),H)
        
new_img_blocks = np.asarray(new_img_blocks)

construct = new_img_blocks.transpose(0,2,1,3).reshape(r,r)
construct = np.asarray(construct)
#compressed_img = compressed_img.resize([h,w])
compressed_img = Image.fromarray(construct,'L')
compressed_img.save("Compressed_picture.jpg",optimize = True,progressive = True)
compressed_img.show()

dec_images = []
dec_img_blocks = [[[] for j in range(p)] for i in range(p)]
for i in range(len(img_blocks)): 
    for j in range(len(img_blocks)):
        dec_img_blocks[i][j] = np.dot(np.dot(H,new_img_blocks[i][j]),H.T)
dec_img_blocks = np.asarray(dec_img_blocks)
reconstruct = dec_img_blocks.transpose(0,2,1,3).reshape(r,r)
reconstruct = np.asarray(reconstruct)
decompressed_img = Image.fromarray(reconstruct)
decompressed_img  = decompressed_img.resize([h,w])
decompressed_img = decompressed_img.convert("L")
decompressed_img.save("Original_picture.jpg")
decompressed_img.show()