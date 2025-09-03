#%% Modules

from typing import Tuple

from abc import ABC

#%% Code

class PositionEncoding(ABC):
    def __init__(self,
                 )->None:
        
        return
    
    def latitude_to_degrees(self,
                            input_string:str,
                            )->float:
        
        splt_val = input_string.split("°")
        degrees = int(splt_val[0])
        splt_val = splt_val[1].split("′")
        arcminutes = int(int(splt_val[0]))
        splt_val = splt_val[1].split("″")
        arcseconds = int(int(splt_val[0]))
        end_str = input_string[-1]

        degrees = degrees + arcminutes/60. + arcseconds/3600.
        if end_str == "S":
            degrees = 360.-degrees
        
        return degrees #degrees*math.pi/180.
    
    def longitude_to_degrees(self,
                             input_string:str,
                             )->float:
        
        splt_val = input_string.split("°")
        degrees = int(splt_val[0])
        splt_val = splt_val[1].split("′")
        arcminutes = int(int(splt_val[0]))
        splt_val = splt_val[1].split("″")
        arcseconds = int(int(splt_val[0]))
        end_str = input_string[-1]

        degrees = degrees + arcminutes/60. + arcseconds/3600.
        if end_str == "W":
            degrees = 360.-degrees
        
        return degrees #degrees*math.pi/180.
    
    def __call__(self,
                 input_string:str,
                 )->Tuple[float, float]:
        
        splt_val = input_string.split(' ')
        
        latitude, longitude = splt_val[0], splt_val[1]
        
        return (
            self.longitude_to_degrees(longitude),
            self.latitude_to_degrees(latitude),
            )
