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
    else:
        print("current fluid not supported")

def plot_phase_diagram(fluid_name : str = "water"):

    PTRIP, TTRIP, PCRIT, TCRIT = get_fluid_data(fluid_name)
    plt.scatter(TTRIP-273.15, PTRIP*1e-5, label='Triple Point', color='red',zorder=10)
    plt.scatter(TCRIT-273.15, PCRIT*1e-5, label='Critical Point', color='green',zorder=10)
    
    p_data_vap = np.geomspace(PTRIP*1.01, PCRIT*0.99999,N)
    T_data_vap = trend.calc_Property_Array("T", "PVAP", p_data_vap.copy(), "", [1 for _ in p_data_vap.copy()], fluid_name, desc_str="Calculating VLE temperatures")

    p_data_sub = np.geomspace(P_MIN, PTRIP*0.99999,N)
    T_data_sub = trend.calc_Property_Array("T", "PSUBV+", p_data_sub.copy(), "", [1 for _ in p_data_sub.copy()], fluid_name, desc_str="Calculating SVE temperatures")

    p_data_mlt = np.geomspace(PTRIP*1.01, P_MAX,N)
    T_data_mlt = trend.calc_Property_Array("T", "PMLTL+", p_data_mlt.copy(), "", [1 for _ in p_data_mlt.copy()], fluid_name, desc_str="Calculating SLE temperatures")

    plt.semilogy([i-273.15 for i in T_data_vap], [i*1e-5 for i in p_data_vap.copy()], label='VLE', color='black')
    plt.semilogy([i-273.15 for i in T_data_sub], [i*1e-5 for i in p_data_sub.copy()], label='SVE', color='black')
    plt.semilogy([i-273.15 for i in T_data_mlt], [i*1e-5 for i in p_data_mlt.copy()], label='SLE', color='black')

    plt.semilogy([TCRIT-273.15, TCRIT-273.15], [0, PCRIT*1e-5], color='gray', linestyle='--')
    plt.semilogy([1.1*(min(T_data_sub)-273.15), TCRIT-273.15], [PCRIT*1e-5, PCRIT*1e-5], color='gray', linestyle='--')
    plt.semilogy([TTRIP-273.15, TTRIP-273.15], [0, PTRIP*1e-5], color='gray', linestyle='--')
    plt.semilogy([1.1*(min(T_data_sub)-273.15), TTRIP-273.15], [PTRIP*1e-5, PTRIP*1e-5], color='gray', linestyle='--')

    plt.text(TTRIP-273.15, PTRIP*1e-5, '  Triple Point S-L-V', color='red', fontsize=10, ha='left', va='top')
    plt.text(TCRIT-273.15, PCRIT*1e-5, '  Critical Point', color='green', fontsize=10, ha='left', va='top')

    plt.text(1.075*(min(T_data_sub)-273.15), 0.9*PTRIP*1e-5, f'{PTRIP:.2f} Pa', color='red', fontsize=10, ha='left', va='top')
    plt.text(1.075*(min(T_data_sub)-273.15), 0.9*PCRIT*1e-5, f'{PCRIT/1e5:.2f} bar', color='green', fontsize=10, ha='left', va='top')

    plt.text(TTRIP-273.15-0.075*(min(T_data_sub)-273.15),2.5*P_MIN/1e5, f'{TTRIP-273.15:.2f} °C \n{TTRIP:.2f} K', color='red', fontsize=10, ha='left', va='top')
    plt.text(TCRIT-273.15-0.075*(min(T_data_sub)-273.15),2.5*P_MIN/1e5, f'{TCRIT-273.15:.2f} °C \n{TCRIT:.2f} K', color='green', fontsize=10, ha='left', va='top')


    plt.xlabel('Temperature (°C)')
    plt.yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    plt.xlim([1.1*(min(T_data_sub)-273.15), 1.1*(max(T_data_vap)-273.15)])
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
    plot_phase_diagram("water")
