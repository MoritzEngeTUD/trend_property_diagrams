import trend_property_diagrams.trend_wrapper as trend
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

P_MIN = 1e-3  # Pa
P_MAX = 210*1e6  # Pa

N = 10000

def get_fluid_data(fluid_name: str):

    if fluid_name == "water":

        PTRIP,TRIP = 611.6,273.16
        PCRIT,TCRIT = 22064000,647.096
        return PTRIP, TRIP, PCRIT, TCRIT
    else:
        print("current fluid not supported")

def plot_phase_diagram(fluid_name : str = "water"):

    PTRIP, TRIP, PCRIT, TCRIT = get_fluid_data(fluid_name)
    plt.scatter(TRIP-273.15, PTRIP*1e-5, label='Triple Point', color='red',zorder=10)
    plt.scatter(TCRIT-273.15, PCRIT*1e-5, label='Critical Point', color='green',zorder=10)
    
    p_data_vap = np.geomspace(PTRIP+1, PCRIT-1,N)
    T_data_vap = trend.calc_Property_Array("T", "PLIQ", p_data_vap.copy(), "", [1 for _ in p_data_vap.copy()], fluid_name, desc_str="Calculating VLE temperatures")

    p_data_sub = np.geomspace(P_MIN, PTRIP-1,N)
    T_data_sub = trend.calc_Property_Array("T", "PSUBS+", p_data_sub.copy(), "", [1 for _ in p_data_sub.copy()], fluid_name, desc_str="Calculating SVE temperatures")

    p_data_mlt = np.geomspace(PTRIP+1, P_MAX,N)
    T_data_mlt = trend.calc_Property_Array("T", "PMLTS+", p_data_mlt.copy(), "", [1 for _ in p_data_mlt.copy()], fluid_name, desc_str="Calculating SLE temperatures")

    plt.semilogy([i-273.15 for i in T_data_vap], [i*1e-5 for i in p_data_vap.copy()], label='VLE', color='blue')
    plt.semilogy([i-273.15 for i in T_data_sub], [i*1e-5 for i in p_data_sub.copy()], label='SVE', color='orange')
    plt.semilogy([i-273.15 for i in T_data_mlt], [i*1e-5 for i in p_data_mlt.copy()], label='SLE', color='purple')
    plt.xlabel('Temperature (°C)')
    plt.yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])

    #plt.gca().set_yticklabels(["1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"], minor=True)
    plt.ylabel('Pressure (bar)')
    plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
    plt.title(f'Phase Diagram for {fluid_name}')
    plt.grid()
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"phase_diagram_{fluid_name}.png", dpi=600)
    plt.savefig(f"phase_diagram_{fluid_name}.pdf")
    plt.savefig(f"phase_diagram_{fluid_name}.svg",transparent=True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("This module is not meant to be run directly. Please import it in your script.")
    plot_phase_diagram("water")
