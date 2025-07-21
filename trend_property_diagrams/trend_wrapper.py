import sys
from pathlib import Path
import importlib.util
from tqdm import tqdm

TREND_DIR = Path("C:/INS/TREND 5.0/")
TREND_DLL_PATH = TREND_DIR.joinpath('TREND_x64.dll')
TREND_ERROR_CODES = TREND_DIR.joinpath('TREND_ERROR_CODES.csv')

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

trend = import_from_path("fluid", TREND_DIR.joinpath("python/fluid.py"))
fluid = trend.fluid

def calc_Property(output : str, Property1 : str, value1 : float, Property2 : str, value2 : float, fluid : str):

    if Property1 in ["P","PLIQ","PVAP","PSUBV+","PSUBS+","PMLTL+","PMLTS+"] : value1 /= 1e6
    elif Property2 in ["P","PLIQ","PVAP","PSUBV+","PSUBS+","PMLTL+","PMLTS+"] : value2 /= 1e6

    inputpair = Property1+Property2

    if type(fluid) is str: fluid = [fluid]

    fld = trend.fluid(inputpair,output,fluid,[1],[1],1,str(TREND_DIR),'specific',str(TREND_DLL_PATH))
    value = fld.TREND_EOS(value1,value2)[0]

    if TREND_ERROR_CODES.exists():
        import csv
        with open(str(TREND_ERROR_CODES), mode='r', encoding='utf-8-sig') as csv_datei:
            reader = csv.reader(csv_datei,delimiter=';')
            error_dict = {zeile[0]: zeile[1] for zeile in reader}
    else : error_dict = {}

    return value if value >= 0 else f"Error-Code : {value} : {error_dict.get(str(int(value)))}"


def calc_Property_Array(output : str, Property1 : str, value1 : list, Property2 : str, value2 : list, fluid : str, desc_str : str = "Calculating properties..."):

    if Property1 in ["P","PLIQ","PVAP","PSUBV+","PSUBS+","PMLTL+","PMLTS+"] : value1 /= 1e6
    elif Property2 in ["P","PLIQ","PVAP","PSUBV+","PSUBS+","PMLTL+","PMLTS+"] : value2 /= 1e6

    inputpair = Property1+Property2

    if type(fluid) is str: fluid = [fluid]

    fld = trend.fluid(inputpair,output,fluid,[1],[1],1,str(TREND_DIR),'specific',str(TREND_DLL_PATH))
    value_pair = [i for i in zip(value1, value2)]
    value_array = [fld.TREND_EOS(values[0], values[1])[0] for values in tqdm(value_pair,desc=desc_str)]

    if TREND_ERROR_CODES.exists():
        import csv
        with open(str(TREND_ERROR_CODES), mode='r', encoding='utf-8-sig') as csv_datei:
            reader = csv.reader(csv_datei,delimiter=';')
            error_dict = {zeile[0]: zeile[1] for zeile in reader}
    else : error_dict = {}

    return [value if value >= 0 else f"Error-Code : {value} : {error_dict.get(str(int(value)))}" for value in value_array]


if __name__ == "__main__":
    print("This module is not meant to be run directly. Please import it in your script.")
