import os
import shutil
import pymeshlab as ml
import time
import gc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ----- Instellingen -----
input_folder = f"copy5000/copy5000"  # originele map

TARGET_VERTICES = 5000
Error = False

# ----- Functie -----
def remeshObject(input_obj):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_obj)
    return ms.current_mesh().vertex_number()

# 📊 Lijsten om resultaten te bewaren
dir_names = []
not_counts = []

# ----- Loop door alle bestanden -----
for root, dirs, files in os.walk(input_folder):
    relative_path = os.path.relpath(root, input_folder)

    Count = 0
    NotCount = 0

    print(f"Map: {relative_path}")

    for file in files:
        if file.lower().endswith(".obj"):
            input_file = os.path.join(root, file)

            if 4000 < remeshObject(input_file) < 10000:
                Count += 1
            else:
                NotCount += 1
                print(f"vertices: {remeshObject(input_file)}")

    if Count + NotCount > 0:
        print(f"Count: {Count}, NotCount: {NotCount}, Percentage: {Count / (Count + NotCount)}")

        # 📊 Resultaat opslaan voor grafiek
        dir_names.append(relative_path)
        not_counts.append(NotCount)

print("Klaar! Alle bestanden zijn gecheckt")
# 📊 Sorteer de resultaten van hoog naar laag op NotCount
sorted_pairs = sorted(zip(dir_names, not_counts), key=lambda x: x[1], reverse=True)
dir_names_sorted, not_counts_sorted = zip(*sorted_pairs)

# 📊 Bar chart maken van NotCount per map met de gevraagde stijl
plt.figure(figsize=(12, 6))
bars = plt.bar(dir_names_sorted, not_counts_sorted, color='#87ceeb')
plt.xticks(rotation=60, ha='right', ticks=range(len(dir_names_sorted)), labels=dir_names_sorted)
plt.xlabel("Shapes")
plt.ylabel("Outliers")
plt.title("Outliers per shape [4900 - 10000 vertices]")

# Zorgen dat y-as alleen hele getallen toont
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Grid zowel horizontaal als verticaal
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()


# Count: 2350, NotCount: 106, Percentage: 0.9568 [4900 - 10000]
# Count: 2282, NotCount: 174, Percentage: 0.9291530944625407 [4900 - 5100]