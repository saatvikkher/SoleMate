import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image, ImageFilter
from util import METADATA

class Sole:
    ''' Class to create a shoeprint object containing metadata (e.g., make, 
    model, size), a dataframe of x,y coordinates, and a dataframe of aligned x,y
    coordinates should the shoe be a "Q" shoe aligned to a "K" shoe.
    '''

    def __init__(self, tiff_path: str) -> None:
        # accessing metadata from csv in util.py
        row = METADATA[METADATA['File Name'] == tiff_path[13:]]

        # metadata fields
        self._file_name = row['File Name'].iloc[0]
        self._scan_method = row['Scan Method'].iloc[0]
        self._model = row['Shoe Make/Model'].iloc[0]
        self._size = row['Shoe Size'].iloc[0]
        self._color = row['Shoe Color'].iloc[0]
        self._foot = row['Foot'].iloc[0]
        self._img = row['Image Number'].iloc[0]
        self._repl = row['Replicate Number'].iloc[0]
        self._visit = row['Visit Number'].iloc[0]
        self._worker = row['Worker Names'].iloc[0]

        # optional aligned coordinates field
        self._aligned = None

        # original coordinates field, set using read tiff helper method
        self._coords = self._read_tiff(tiff_path)
    
    def __str__(self):
        return "Shoeprint Object: " + self.file_name

    @property
    def file_name(self) -> str:
        '''Getter method for shoeprint file_name'''
        return self._file_name
    
    @property
    def scan_method(self) -> str:
        '''Getter method for shoeprint scan method'''
        return self._scan_method

    @property
    def model(self) -> str:
        '''Getter method for shoe model'''
        return self._model

    @property
    def size(self) -> str:
        '''Getter method for shoe size'''
        return self._size

    @property
    def color(self) -> str:
        '''Getter method for shoe color'''
        return self._color

    @property
    def foot(self) -> str:
        '''Getter method for shoe foot (left/right)'''
        return self._foot

    @property
    def img(self):
        '''Getter method for shoeprint scan image number'''
        return self._img

    @property
    def repl(self):
        '''Getter method for shoeprint scan replicate number'''
        return self._repl

    @property
    def visit(self):
        '''Getter method for shoeprint scan visit number'''
        return self._visit

    @property
    def worker(self) -> str:
        '''Getter method for worker who scanned shoe'''
        return self._worker

    @property
    def aligned(self) -> pd.DataFrame:
        '''Getter method for dataframe of aligned shoeprint coordinates'''
        return self._aligned

    @aligned.setter
    def aligned(self, value) -> None:
        '''Setter method for dataframe of aligned shoeprint coordinates'''
        try:
            self._aligned = value
        except Exception as e:
            print("Must be a pandas DataFrame:", str(e))

    @property
    def coords(self) -> pd.DataFrame:
        '''Getter method for dataframe of original shoeprint coordinates'''
        return self._coords


    def _read_tiff(self, link: str) -> pd.DataFrame:
        '''Helper method which takes an image's file address and converts it 
        to a set of x,y coordinates using edge detection
        '''

        # open image and convert to a grayscale numpy array
        img = Image.open(link)
        img_arr = np.array(img)
        img = img.convert("L")

        # edge detection
        img = img.filter(ImageFilter.FIND_EDGES)
        
        # invert colors to light background, dark dots
        inv = PIL.ImageOps.invert(img)

        # extract image dimensions and crop image
        image_height, image_width = np.array(inv).shape
        border_width = 160 # CHANGE TO NOT HARD-CODED
        crop = inv.crop((border_width, border_width, image_width-border_width, 
                         image_height-border_width))
        crop_arr = np.array(crop)

        # change from grayscale to binary black/white to create coordinates df
        crop_arr = crop_arr < 127
        rows, cols = np.where(crop_arr)
        df = pd.DataFrame({"x": rows, "y": cols})
        
        return df
    
    def plot(self, color = "red", size: float = 0.5):

        # Plot scatter plot 1
        plt.scatter(self.coords.x, self.coords.y, color=color, s = size)

        # Adjust the plot boundaries
        plt.tight_layout()

        plt.show()
        
if __name__ == "__main__":
    pass