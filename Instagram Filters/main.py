# Modules
import cv2 as cv
import numpy as np
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass

# I add this class as a note to show you what is output type of methods in 
# "InstagramFilters" and it's not necessary for code and if you want you can delete it.
@dataclass
class Image:
    pass

class InstagramFilters():
    def __init__(self, image_path: str) -> Image:
        self.image_path = image_path

    def grayscale(self):
        # Passing zero to get a grayscale image.
        gray_image = cv.imread(self.image_path,0)
        return gray_image
    
    def brightness(self, alpha=0, beta=1) -> Image :
        # Using normalize function for changing image brightness. 
        image =  cv.imread(self.image_path)
        image = cv.normalize(image, image.copy(), alpha, beta,
        cv.NORM_MINMAX, dtype=cv.CV_32F)
        return image
    
    def sharp(self) -> Image:
        image =  cv.imread(self.image_path)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        # Sharpen the image.
        sharpened_image = cv.filter2D(image, -1, kernel) 
        return sharpened_image
    
    def pencil(self,sigma_s=40,sigma_r=0.06) -> Image:
        # With pencilSketch function we can make image like drawn by pencil with do type of colors.
        image = cv.imread(self.image_path)
        pencil, _  = cv.pencilSketch(
        image,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        shade_factor=0.05
        )
        return pencil
    
    def sketch(self,sigma_s=40,sigma_r=0.06) -> Image:
        # Using pencilSketch but with different output.
        image = cv.imread(self.image_path)
        _, sketch  = cv.pencilSketch(
        image,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        shade_factor=0.05
        )
        return sketch

    def detile_enhance(self,sigma_s=10,sigma_r=0.15) -> Image:
        # Filter to have more details.
        image = cv.imread(self.image_path)
        enhanced = cv.detailEnhance(
        image,
        sigma_s=sigma_s,
        sigma_r=sigma_r
        )
        return enhanced
    
    def clahe(self,clipLimit=2.0,tileGridSize=(8,8)) -> Image:
        image = cv.imread(self.image_path)
        # First get all of color channels for our image.
        channels = cv.split(image)
        # Create a clahe object.
        clahe = cv.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
        # Use 'apply' method on channels and put them in list.
        clahe_channels = []
        for channel in channels :
            clahe_channels.append(clahe.apply(channel))
        # Merge channels and get the final image.
        final_image = cv.merge(clahe_channels)
        
        return final_image
        
    def LookupTable(self,x, y) -> Image:
        # Creating a LookupTable(lut) for "warmer_img" function.
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    def warmer_img(self,thersh_hold: int) -> Image:
        # Getting all of scales needed in warm function.
        scale1 = (((64-32) * thersh_hold ) / 100 ) + 32
        scale2 = (((128-64) * thersh_hold ) / 100 ) + 64
        scale3 = (((80-40) * thersh_hold ) / 100 ) + 40
        scale4 = (((160-80) * thersh_hold ) / 100 ) + 80
        image = cv.imread(self.image_path)
        # Getting all needed.
        scales = [int(scale1),int(scale2),int(scale3),int(scale4)]
        # Creating luts.
        increaseLookupTable = self.LookupTable([0, scales[0], scales[1], 256], [0, scales[2], scales[3], 256])
        decreaseLookupTable = self.LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        # Spiliting image channels.
        blue_channel, green_channel,red_channel  = cv.split(image)
        # Useing luts for Red and Blue channels.
        red_channel = cv.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        # Combining channles.
        image = cv.merge((blue_channel, green_channel, red_channel))
        return image

    def colder_img(self, thersh_hold: int) -> Image:
        # Getting all of scales needed in cold function.
        scale1 = (((75-45) * thersh_hold ) / 100 ) + 45
        scale2 = (((35-10) * thersh_hold ) / 100 ) + 10
        image = cv.imread(self.image_path)
        # Spiliting image channels.
        scale1 , scale2 =  [int(scale1),int(scale2)]
        blue_channel, green_channel,red_channel = cv.split(image)
        blue_channel = cv.add(blue_channel,scale1)
        red_channel = cv.subtract(red_channel,scale2)
        # Combining color channles.
        image = cv.merge((blue_channel, green_channel, red_channel))
        return image
    
    def invert(self) -> Image:
        # Invert filter.
        image = cv.imread(self.image_path)
        inv = cv.bitwise_not(image)
        return inv

    def stylization(self,sigma_s=15,sigma_r=0.55) -> Image:
        # Stylization filter.
        image = cv.imread(self.image_path)
        image = cv.stylization(image,sigma_s=sigma_s,sigma_r=sigma_r)
        return image
    
    def Sepia(self) -> Image:
        image = cv.imread(self.image_path)
        # Convert the image into float type to prevent loss during operations.
        image_float = np.array(image, dtype=np.float64) 
        # Split the blue, green, and red channel of the image.
        blue_channel, green_channel, red_channel = cv.split(image_float)
        # Apply the Sepia filter by perform the matrix multiplication between 
        # the image and the sepia matrix.
        output_blue = (red_channel * .272) + (green_channel *.534) + (blue_channel * .131)
        output_green = (red_channel * .349) + (green_channel *.686) + (blue_channel * .168)
        output_red = (red_channel * .393) + (green_channel *.769) + (blue_channel * .189)
        # Merge the blue, green, and red channel.
        output_image = cv.merge((output_blue, output_green, output_red)) 
        # Set the values > 255 to 255.
        output_image[output_image > 255] = 255
        # Convert the image back to uint8 type.
        output_image =  np.array(output_image, dtype=np.uint8)
        return output_image
