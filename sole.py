import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image, ImageFilter
from utils.util import BLACK_WHITE_THRESHOLD, WILLIAMS_PURPLE

class Sole:
    '''
    Sole is a class to create a shoeprint object. A shoeprint object holds the
    shoeprint point-cloud of a shoeprint. 
    
    Shoeprints may be read in as pandas DataFrames or as image files (tiff, 
    jpg, etc.). Shoeprint images will be filtered using a selected grayscale 
    threshold. The Sole will then be processed using PIL's edge-detection. Both 
    the image of points selected by edge-detection and the full shoeprint image 
    will be saved as point-clouds. Shoes processed as dataframes will skip the 
    edge detection process.
    '''

    def __init__(self,
                 image_path: str = None,
                 border_width: int = 1,
                 is_image: bool = True,
                 coords: pd.DataFrame = None,
                 flipped: bool = False) -> None:
        '''
        Initialization function for the Sole class.

        Inputs:
            image_path (str): the location for the shoe to be read in
            border_width (int): the crop width if there is a border around the
                shoeprint image
            is_image (bool): if the shoeprint is not an image file, it can be
                read-in directly by its coords. Edge detection will be skipped.
            coords (pd.DataFrame): the point-cloud for a non-image shoeprint
            flipped (bool): whether or not a shoeprint needs to be reversed
        '''

        self._aligned_coordinates = None
        self._aligned_full = None

        if is_image:
            self._coords, self._coords_full = self._image_to_coords(image_path, 
                                                                   border_width)
        else:
            self._coords = coords
            self._coords_full = coords

        if flipped:
            self._coords = self._flip_coords(self._coords)
            self._coords_full = self._flip_coords(self._coords_full)

    @property
    def aligned_coordinates(self) -> pd.DataFrame:
        '''Getter method for dataframe of aligned shoeprint coordinates'''
        return self._aligned_coordinates

    @aligned_coordinates.setter
    def aligned_coordinates(self, value) -> None:
        '''Setter method for dataframe of aligned shoeprint coordinates'''
        try:
            self._aligned_coordinates = value
        except Exception as e:
            print("Must be a pandas DataFrame:", str(e))

    @property
    def aligned_full(self) -> pd.DataFrame:
        '''Getter method for dataframe of aligned shoeprint coordinates'''
        return self._aligned_full

    @aligned_full.setter
    def aligned_full(self, value) -> None:
        '''Setter method for dataframe of aligned shoeprint coordinates'''
        try:
            self._aligned_full = value
        except Exception as e:
            print("Must be a pandas DataFrame:", str(e))

    @property
    def coords(self) -> pd.DataFrame:
        '''Getter method for dataframe of original shoeprint coordinates'''
        return self._coords

    @coords.setter
    def coords(self, value) -> None:
        '''Setter method for dataframe of coordinates'''
        try:
            self._coords = value
        except Exception as e:
            print("Must be a pandas DataFrame:", str(e))

    @property
    def coords_full(self) -> pd.DataFrame:
        '''Getter method for dataframe of original shoeprint coordinates'''
        return self._coords_full

    @coords_full.setter
    def coords_full(self, value) -> None:
        '''Setter method for dataframe of coordinates'''
        try:
            self._coords_full = value
        except Exception as e:
            print("Must be a pandas DataFrame:", str(e))

    def _image_to_coords(self, link: str, border_width: int) -> pd.DataFrame:
        '''
        Helper method which takes an image's file address and converts it 
        to a set of x,y coordinates using edge detection. If the image has a
        border, setting border_width allows image cropping.

        Inputs:
            link (str): the location of the image
            border_width (int): gets rid of this amount of pixels around border
        
        Returns:
            pd.Dataframe: the coordinates of the image
        '''
        # open image and convert to a grayscale numpy array
        img = Image.open(link)
        img = img.convert("L")
        img_full = img.convert("L")

        # edge detection
        img = img.filter(ImageFilter.FIND_EDGES)

        # invert colors to light background, dark dots
        inv = PIL.ImageOps.invert(img)

        # extract image dimensions and crop image
        image_height, image_width = np.array(inv).shape

        # fix issue of the image border being treated as an edge
        if border_width == 0:
            crop = inv.crop((1, 1, image_width-1, image_height-1))
        else:     
            crop = inv.crop((border_width, border_width, image_width-border_width, image_height-border_width))
        crop_arr = np.array(crop)
        
        # crop image for the non-edge-detection shoeprint
        crop_full = img_full.crop((border_width, border_width, 
                           image_width-border_width, image_height-border_width))
        crop_arr_full = np.array(crop_full)

        # change from grayscale to binary black/white to create coordinates df
        crop_arr = crop_arr < BLACK_WHITE_THRESHOLD
        crop_arr_full = crop_arr_full < BLACK_WHITE_THRESHOLD
        rows, cols = np.where(crop_arr)
        rows_full, cols_full = np.where(crop_arr_full)
        df = pd.DataFrame({"x": rows, "y": cols})
        df_full = pd.DataFrame({"x": rows_full, "y": cols_full})

        return df, df_full

    def plot(self, color=WILLIAMS_PURPLE, size: float = 0.1, filename = None):
        '''
        Plots Sole based on original coordinates.

        Inputs:
            color (str): defaults to Williams College Purple #500082
            size (float): size of plot points s=size in plt.scatter()
            filename (str): output file to save fig if desired, if no input, fig
                will not be saved

        Returns:
            None
        '''
        plt.scatter(self.coords.x, self.coords.y, color=color, s=size)
        plt.tight_layout()

        # keep aspect ratio the same so that the shoeprint is not stretched
        plt.gca().set_aspect('equal')

        if filename is not None:
            plt.savefig(filename, dpi=600)

        plt.show()

    def _flip_coords(self, coords):
        '''
        Flips the coordinates of the shoe along the x-axis. This is used if an 
        image is reflected.
        '''
        temp_coords = coords.copy(deep=True)
        max_y = max(coords['y'])
        temp_coords['y'] = temp_coords['y']*-1 + max_y
        return temp_coords