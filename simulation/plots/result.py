import pandas as pd
import matplotlib, os
import matplotlib.pyplot as plt

matplotlib.use('Agg')
input_csv1 = pd.read_csv("./Entanglement fidelity _3.csv")
input_csv2 = pd.read_csv("./Entanglement fidelity with filtering_1.csv")
input_csv3 = pd.read_csv("./Entanglement fidelity with distil_1.csv")
input_csv4 = pd.read_csv("./Entanglement fidelity with distil & filtering_2.csv")

input_csv11 = pd.read_csv("./Teleportation fidelity_1.csv")
input_csv22 = pd.read_csv("./Teleportation fidelity with filtering_1.csv")
input_csv33 = pd.read_csv("./Teleportation fidelity with distil_1.csv")
input_csv44 = pd.read_csv("./Teleportation fidelity with distil & filtering_2.csv")

data1 = input_csv1.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data2 = input_csv2.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data3 = input_csv3.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data4 = input_csv4.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()

data11 = input_csv11.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data22 = input_csv22.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data33 = input_csv33.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data44 = input_csv44.groupby("depolar_rate")['F2'].agg(fidelity='mean', sem='sem').reset_index()

xcolumn = data1['depolar_rate']

save_dir = "."
existing_files1 = len([f for f in os.listdir(save_dir) if f.startswith("Result_entanglement")])
existing_files2 = len([f for f in os.listdir(save_dir) if f.startswith("Result_teleportation")])
filename1 = f"{save_dir}/Result_entanglement_{existing_files1 + 1}.png"
filename2 = f"{save_dir}/Result_teleportation_{existing_files2 + 1}.png"

plt.figure()
plt.errorbar(xcolumn, data1['fidelity'], yerr=data1['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data2['fidelity'], yerr=data2['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data3['fidelity'], yerr=data3['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data4['fidelity'], yerr=data4['sem'], marker="o", label='Distil&Filtering')

plt.xlabel('depolar_rate')
plt.ylabel('Fidelity')
plt.title('Fidelity of entanglement')
plt.legend()
plt.grid()

plt.savefig(filename1)
print(f"Plot saved as {filename1}")

plt.figure()
plt.errorbar(xcolumn, data11['fidelity'], yerr=data1['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data22['fidelity'], yerr=data2['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data33['fidelity'], yerr=data3['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data44['fidelity'], yerr=data4['sem'], marker="o", label='Distil&Filtering')

plt.xlabel('depolar_rate')
plt.ylabel('Fidelity')
plt.title('Fidelity of teleportation')
plt.legend()
plt.grid()

plt.savefig(filename2)
print(f"Plot saved as {filename2}")