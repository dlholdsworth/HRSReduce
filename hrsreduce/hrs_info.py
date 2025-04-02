from datetime import datetime,timedelta
import logging

logger = logging.getLogger(__name__)

class Instrument:

    def __init__(self):
        #:str: Name of the instrument (lowercase)
        self.name = self.__class__.__name__.lower()
        #:dict: Information about the instrument
        self.info = self.load_info()


    def load_info(self):
        """
        Load static instrument information
        Either as fits header keywords or static values

        Returns
        ------
        info : dict(str:object)
            dictionary of REDUCE names for properties to Header keywords/static values
        """
        # Tips & Tricks:
        # if several modes are supported, use a list for modes
        # if a value changes depending on the mode, use a list with the same order as "modes"
        # you can also use values from this dictionary as placeholders using {name}, just like str.format

        this = os.path.dirname(__file__)
        fname = f"{self.name}.json"
        fname = os.path.join(this, fname)
        with open(fname) as f:
            info = json.load(f)
        return info
        

    def sort_files_2(self, input_dir, *args,**kwargs):
        """
        Sort a set of fits files into different categories
        types are: bias, flat, wavecal, orderdef, spec

        Parameters
        ----------
        input_dir : str
            input directory containing the files to sort
        night : str
            observation night, possibly with wildcards
        mode : str
            instrument mode
        arm : str
            instrument arm
            
        Returns
        -------
        files_per_night : list[dict{str:dict{str:list[str]}}]
            a list of file sets, one entry per night, where each night consists of a dictionary with one entry per setting,
            each fileset has five lists of filenames: "bias", "flat", "order", "wave", "spec", organised in another dict
        nights_out : list[datetime]
            a list of observation times, same order as files_per_night
        """
        
        files = self.find_files(input_dir)
        bias_files = []
        flat_files = []
        arc_files = []
        lfc_files = []
        sci_files = []

        for file in files:
            with fits.open(file) as hdul:
                hdr = hdul[0].header
                if hdr[self.info['kw_bias']] == self.info['id_bias']:
                    bias_files.append(file)
                    continue
                if hdr[self.info['kw_modes']] == kwargs['mode']:
                    if hdr[self.info['kw_flat']] == self.info['id_flat']:
                        flat_files.append(file)
                    elif hdr[self.info['kw_wave']] == self.info['id_wave']:
                        arc_files.append(file)
                    elif hdr[self.info['kw_spec']] == self.info['id_spec']:
                        sci_files.append(file)
                    elif hdr[self.info['kw_comb']] == self.info['id_comb']:
                        lfc_files.append(file)
                    else:
                        logger.warning("File %s does not match and expected in %s, %s, %s or %s", file,self.info['id_flat'],self.info['id_wave'],self.info['id_spec'],self.info['id_comb'] )
                    
        return bias_files,flat_files,arc_files,lfc_files,sci_files


