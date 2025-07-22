import trend_property_diagrams.trend_wrapper as trend
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

P_MIN = 1e-3  # Pa
P_MAX = 210*1e6  # Pa

N = 100

def get_fluid_data(fluid_name: str):

    if fluid_name == "water":

        PTRIP,TTRIP = 611.66,273.16
        PCRIT,TCRIT = 22064000,647.096
        return PTRIP, TTRIP, PCRIT, TCRIT
    elif fluid_name == "CO2":

        PTRIP,TTRIP = 517964,216.592
        PCRIT,TCRIT = 7377297,304.128
        return PTRIP, TTRIP, PCRIT, TCRIT
    else: print("current fluid not supported")

def extract_data(data,):
    ret_ar = {}
    if data and isinstance(data[0], dict):
        for key in data[0].keys():
            # Extrahieren der Werte für jeden Schlüssel aus allen Dictionaries
            ret_ar[key] = np.array([d.get(key, np.nan) if isinstance(d, dict) else np.nan for d in data])
    return ret_ar  
     
def get_VLE_dome(fluid_name: str):
    PTRIP, TTRIP, PCRIT, TCRIT = get_fluid_data(fluid_name)

    p_data_vle = np.geomspace(PTRIP*1.01, PCRIT*0.99999, N)
    data_vle_vap = trend.calc_ALL_Property_Array("PVAP", p_data_vle.copy(), "", [1 for _ in p_data_vle.copy()], fluid_name, desc_str="Calculating VLE Vapor Data...")
    data_vle_liq = trend.calc_ALL_Property_Array("PLIQ", p_data_vle.copy(), "", [1 for _ in p_data_vle.copy()], fluid_name, desc_str="Calculating VLE Liquid Data...")
    
    return extract_data(data_vle_liq), extract_data(data_vle_vap)

def get_SVE_dome(fluid_name: str):
    PTRIP, TTRIP, PCRIT, TCRIT = get_fluid_data(fluid_name)

    p_data_sve = np.geomspace(P_MIN,PTRIP*0.99 , N)
    data_sve_vap = trend.calc_ALL_Property_Array("PSUBV+", p_data_sve.copy(), "", [1 for _ in p_data_sve.copy()], fluid_name, desc_str="Calculating SVE Vapor Data...")
    data_sve_sol = trend.calc_ALL_Property_Array("PSUBS+", p_data_sve.copy(), "", [1 for _ in p_data_sve.copy()], fluid_name, desc_str="Calculating SVE Solid Data...")

    return extract_data(data_sve_sol),extract_data(data_sve_vap)

def plot_phase_diagram(fluid_name : str = "water"):

    PTRIP, TTRIP, PCRIT, TCRIT = get_fluid_data(fluid_name)
    plt.scatter(TTRIP-273.15, PTRIP*1e-5, label='Triple Point', color='red',zorder=10)
    plt.scatter(TCRIT-273.15, PCRIT*1e-5, label='Critical Point', color='green',zorder=10)
    
    #liq_arrays_vle,vap_arrays_vle = get_VLE_dome(fluid_name)
    sol_arrays_sve,vap_arrays_sve = get_SVE_dome(fluid_name)

    #plt.semilogy(vap_arrays_vle["T"]-273.15,vap_arrays_vle["P"]*10, label='Evaporation', color='black') # p : MPa in bar
    #plt.semilogy(liq_arrays_vle["T"]-273.15,liq_arrays_vle["P"]*10, label='Condensation', color='black')

    plt.semilogy(sol_arrays_sve["T"]-273.15,sol_arrays_sve["P"]*10, label='Sublimation', color='blue')
    plt.semilogy(vap_arrays_sve["T"]-273.15,vap_arrays_sve["P"]*10, label='Resublimation', color='blue')

    plt.text(TTRIP-273.15, PTRIP*1e-5, '  Triple Point S-L-V', color='red', fontsize=10, ha='left', va='top')
    plt.text(TCRIT-273.15, PCRIT*1e-5, '  Critical Point', color='green', fontsize=10, ha='left', va='top')

    plt.xlabel('Temperature (°C)')
    plt.xlim(-100,400)
    plt.ylabel('Pressure (bar)')
    plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
    plt.title(f'Phase Diagram for {fluid_name}')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"phase_diagram_{fluid_name}.png", dpi=600)
    plt.savefig(f"phase_diagram_{fluid_name}.pdf")
    plt.savefig(f"phase_diagram_{fluid_name}.svg",transparent=True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("This module is not meant to be run directly. Please import it in your script.")
    plot_phase_diagram("Water")