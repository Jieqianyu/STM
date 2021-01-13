import re
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os
import random
from functools import reduce

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

paths = []
while True:
    path = input()
    if not path:
        break
    paths.append(path)
    print("Paths: ", paths)

rets = []
pattern = re.compile(r'.*Loss: (\d+.\d+)')
for path in paths:
    with open(path) as f:
        txt_str = f.read()
    ret = re.findall(pattern, txt_str)
    ret = [float(x) for x in ret]
    rets.append(ret)
for ret in rets:
    name = random.choice(sorted_names)
    # print(name)
    plt.plot(ret, color=colors[name])
label = [os.path.basename(path).split('.')[0] for path in paths]
plt.legend(label, loc='upper left')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()
figure_name = reduce(lambda x, y: x+y, label)
print(figure_name)
# plt.savefig('figures/'+ figure_name + '.jpg')
