import time
import matplotlib.pyplot as plt

thickness = 0.00008 # thickness of paper
folded_thickness = thickness * 2**43
print("Thickness: {} meters".format(folded_thickness))
print("Thickness: {: .2f} kilometers".format(folded_thickness/1000))

folded_thickness = thickness*2

folded_thickness_list = []
folded_thickness = thickness*2
print(type(folded_thickness_list))
for i in range(44):
  folded_thickness_list.append(folded_thickness)
  folded_thickness = folded_thickness * 2
print("lenght of list:{}".format(len(folded_thickness_list)))
print(folded_thickness_list)

plt.title("Thickness of folded paper")
plt.xlabel("number of folds")
plt.ylabel("thickness[m]")
plt.tick_params(labelsize=14) # Make settings related to axis values
plt.plot(folded_thickness_list, color='green', linewidth=3, linestyle='dashed')
plt.show()