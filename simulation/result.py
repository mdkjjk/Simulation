# シミュレーション結果を1つのグラフに表示するためのプログラム

import pandas as pd
import matplotlib, os
import matplotlib.pyplot as plt

matplotlib.use('Agg')
input_csv1 = pd.read_csv("./plots_clean/sfidelity/Original_Entanglement fidelity_2.csv")
input_csv2 = pd.read_csv("./plots_clean/sfidelity/Filtering_Entanglement fidelity_2.csv")
input_csv3 = pd.read_csv("./plots_clean/sfidelity/Distil_Entanglement fidelity_2.csv")
input_csv4 = pd.read_csv("./plots_clean/sfidelity/Distil&Filtering_Entanglement fidelity_2.csv")
input_csv5 = pd.read_csv("./plots_clean/sfidelity/Filtering2_Entanglement fidelity_2.csv")

input_csv11 = pd.read_csv("./plots_clean/sfidelity/Original_Teleportation fidelity_2.csv")
input_csv22 = pd.read_csv("./plots_clean/sfidelity/Filtering_Teleportation fidelity_2.csv")
input_csv33 = pd.read_csv("./plots_clean/sfidelity/Distil_Teleportation fidelity_2.csv")
input_csv44 = pd.read_csv("./plots_clean/sfidelity/Distil&Filtering_Teleportation fidelity_2.csv")
input_csv55 = pd.read_csv("./plots_clean/sfidelity/Filtering2_Teleportation fidelity_2.csv")

data1 = input_csv1.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data2 = input_csv2.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data3 = input_csv3.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data4 = input_csv4.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data5 = input_csv5.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()

data11 = input_csv11.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data22 = input_csv22.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data33 = input_csv33.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data44 = input_csv44.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()
data55 = input_csv55.groupby("source_fidelity")['F2'].agg(fidelity='mean', sem='sem').reset_index()

time1 = input_csv11.groupby("source_fidelity")['time'].agg(time='mean', sem='sem').reset_index()
time2 = input_csv22.groupby("source_fidelity")['time'].agg(time='mean', sem='sem').reset_index()
time3 = input_csv33.groupby("source_fidelity")['time'].agg(time='mean', sem='sem').reset_index()
time4 = input_csv44.groupby("source_fidelity")['time'].agg(time='mean', sem='sem').reset_index()
time5 = input_csv55.groupby("source_fidelity")['time'].agg(time='mean', sem='sem').reset_index()

xcolumn = data11['source_fidelity']

save_dir = "./plots_clean/sfidelity"
existing_files1 = len([f for f in os.listdir(save_dir) if f.startswith("Result_entanglement")])
existing_files2 = len([f for f in os.listdir(save_dir) if f.startswith("Result_teleportation")])
existing_files3 = len([f for f in os.listdir(save_dir) if f.startswith("Result_time")])

filename1 = f"{save_dir}/Result_entanglement_{existing_files1 + 1}.png"
filename2 = f"{save_dir}/Result_teleportation_{existing_files2 + 1}.png"
filename3 = f"{save_dir}/Result_time_{existing_files3 + 1}.png"

plt.figure()
plt.errorbar(xcolumn, data1['fidelity'], yerr=data1['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data2['fidelity'], yerr=data2['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data3['fidelity'], yerr=data3['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data4['fidelity'], yerr=data4['sem'], marker="o", label='Distil&Filtering')
plt.errorbar(xcolumn, data5['fidelity'], yerr=data5['sem'], marker="o", label='Filtering x2')


plt.xlabel('source_fidelity')
plt.ylabel('Fidelity')
plt.title('Fidelity of entanglement')
plt.legend()
plt.grid()

plt.savefig(filename1)
print(f"Plot saved as {filename1}")

plt.figure()
plt.errorbar(xcolumn, data11['fidelity'], yerr=data11['sem'], marker="o", label='Original')
plt.errorbar(xcolumn, data22['fidelity'], yerr=data22['sem'], marker="o", label='Filtering')
plt.errorbar(xcolumn, data33['fidelity'], yerr=data33['sem'], marker="o", label='Distil')
plt.errorbar(xcolumn, data44['fidelity'], yerr=data44['sem'], marker="o", label='Distil&Filtering')
plt.errorbar(xcolumn, data55['fidelity'], yerr=data55['sem'], marker="o", label='Filtering x2')

plt.xlabel('source_fidelity')
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
plt.errorbar(xcolumn, time5['time'], yerr=time5['sem'], marker="o", label='Filtering x2')

plt.xlabel('source_fidelity')
plt.ylabel('Time [ns]')
plt.title('Average time')
plt.legend()
plt.grid()

plt.savefig(filename3)
print(f"Plot saved as {filename3}")