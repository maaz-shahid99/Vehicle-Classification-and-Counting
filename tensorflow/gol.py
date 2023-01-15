from PIL import Image
import os
import glob

files = glob.glob("C:/Users/Maaz Shahid/Desktop/TF2/Tensorflow/workspace/images/collectedimages/MotorCycle/*")
len(files)


for file in files:
    im = Image.open(file)
    rgb_im = im.convert("RGB")
    str=file.split("\\",1)
    rgb_im.save('C:/Users/Maaz Shahid/Desktop/Pic/'+str[1], 'JPEG')

    #rgb_im.save("Desktop/Pic")

#im = Image.open(r"C:/Users/Maaz Shahid/Desktop/TF2/TFODCourse/Tensorflow/workspace/images/collectedimages/LightVehicle/lotus-elise-s-model-year-2006-yellow-driving-diagonal-from-the-back-B4CW41.jpeg")
#print("The size of the image before conversion : ", end = "")
#print(os.path.getsize("lotus-elise-s-model-year-2006-yellow-driving-diagonal-from-the-back-B4CW41.jpeg"))
  
# converting to jpg
#rgb_im = im.convert("RGB")
  
# exporting the image
#rgb_im.show()

#rgb_im.save("001.jpeg")
#print("The size of the image after conversion : ", end = "")
#print(os.path.getsize("lotus-elise-s-model-year-2006-yellow-driving-diagonal-from-the-back-B4CW41.jpeg"))