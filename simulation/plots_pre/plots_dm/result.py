import pandas as pd
import matplotlib, os
import matplotlib.pyplot as plt

matplotlib.use('Agg')
"""input_csv1 = pd.read_csv("./Original_Entanglement fidelity_2.csv")
input_csv2 = pd.read_csv("./Filtering_Entanglement fidelity_4.csv")
input_csv3 = pd.read_csv("./Distil_Entanglement fidelity_2.csv")
input_csv4 = pd.read_csv("./Distil&Filtering_Entanglement fidelity_4.csv")"""

input_csv11 = pd.read_csv("./Original_Teleportation fidelity_4.csv")
input_csv22 = pd.read_csv("./Filtering_Teleportation fidelity_4.csv")
input_csv33 = pd.read_csv("./Distil_Teleportation fidelity_2.csv")
input_csv44 = pd.read_csv("./Distil&Filtering_Teleportation fidelity_6.csv")

"""data1 = input_csv1.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data2 = input_csv2.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data3 = input_csv3.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data4 = input_csv4.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()"""

data11 = input_csv11.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data22 = input_csv22.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data33 = input_csv33.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data44 = input_csv44.groupby("node_distance")['F2'].agg(fidelity='mean', sem='sem').reset_index()

time1 = input_csv11.groupby("node_distance")['time'].agg(time='mean', sem='sem').reset_index()
time2 = input_csv22.groupby("node_distance")['time'].agg(time='mean', sem='sem').reset_index()
time3 = input_csv33.groupby("node_distance")['time'].agg(time='mean', sem='sem').reset_index()
time4 = input_csv44.groupby("node_distance")['time'].agg(time='mean', sem='sem').reset_index()

xcolumn = data11['node_distance']

save_dir = "."
existing_files1 = len([f for f in os.listdir(save_dir) if f.startswith("Result_entanglement")])
existing_files2 = len([f for f in os.listdir(save_dir) if f.startswith("Result_teleportation")])
existing_files3 = len([f for f in os.listdir(save_dir) if f.startswith("Result_time")])

filename1 = f"{save_dir}/Result_entanglement_{existing_files1 + 1}.png"
filename2 = f"{save_dir}/Result_teleportation_{existing_files2 + 1}.png"
filename3 = f"{save_dir}/Result_time_{existing_files3 + 1}.png"

"""plt.figure()
plt.errorbar(xcolumn, data1['fidelity'], yerr=data1['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data2['fidelity'], yerr=data2['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data3['fidelity'], yerr=data3['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data4['fidelity'], yerr=data4['sem'], marker="o", label='Distil&Filtering')

plt.xlim(0, 42)
plt.ylim(bottom = 0.7)
plt.xlabel('node_distance')
plt.ylabel('Fidelity')
plt.title('Fidelity of entanglement')
plt.legend()
plt.grid()

plt.savefig(filename1)
print(f"Plot saved as {filename1}")"""

plt.figure()
plt.errorbar(xcolumn, data11['fidelity'], yerr=data11['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data22['fidelity'], yerr=data22['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data33['fidelity'], yerr=data33['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data44['fidelity'], yerr=data44['sem'], marker="o", label='Distil&Filtering')

plt.xlabel('node_distance')
plt.ylabel('Fidelity')
plt.title('Fidelity of teleportation')
plt.legend()
plt.grid()

plt.savefig(filename2)
print(f"Plot saved as {filename2}")

plt.figure()
plt.errorbar(xcolumn, time1['time'], yerr=time1['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, time2['time'], yerr=time2['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, time3['time'], yerr=time3['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, time4['time'], yerr=time4['sem'], marker="o", label='Distil&Filtering')

plt.xlabel('node_distance')
plt.ylabel('Time [ns]')
plt.title('Average time')
plt.legend()
plt.grid()

plt.savefig(filename3)
print(f"Plot saved as {filename3}")