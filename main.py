import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
#right data
trues = [
[[0,1,1,1,0,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[0,1,1,1,0,0]],
[[0,0,1,0,0,0],
[1,1,1,0,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[1,1,1,1,1,0]],
[[1,1,1,1,0,0],
[1,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,1,0,0],
[0,0,1,0,0,0],
[0,1,0,0,0,0],
[1,0,0,0,0,0],
[1,1,1,1,1,0]],
[[1,1,1,1,0,0],
[1,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,1,0,0],
[0,0,1,1,0,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[1,1,1,1,0,0]],
[[0,0,0,0,1,0],
[0,0,0,1,1,0],
[0,0,1,0,1,0],
[0,1,0,0,1,0],
[1,1,0,0,1,0],
[1,1,1,1,1,1],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0]],
[[1,1,1,1,1,0],
[1,0,0,0,0,0],
[1,0,0,0,0,0],
[1,1,1,1,0,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[1,0,0,0,1,0],
[1,1,1,1,0,0]],
[[0,0,1,1,1,0],
[0,1,0,0,0,0],
[1,0,0,0,0,0],
[1,1,1,1,0,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[0,1,1,1,0,0]],
[[1,1,1,1,1,1],
[0,0,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,1,0,0],
[0,0,0,1,0,0],
[0,0,1,0,0,0],
[0,0,1,0,0,0],
[0,1,0,0,0,0],
[1,1,0,0,0,0]],
[[0,1,1,1,0,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,1,0,0,1,0],
[0,1,1,1,0,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[0,1,1,1,0,0]],
[[0,1,1,1,0,0],
[1,0,0,1,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[1,0,0,0,1,0],
[0,1,1,1,1,0],
[0,0,0,0,1,0],
[0,0,0,1,0,0],
[1,1,1,0,0,0]]]

def detect(number_array):
	maxx = None
	val = None
	tempo = number_array
	tempo.pop(0)
	trues_np = np.array(trues)
	tempo = np.array(tempo)
	count = 0
	for array in trues_np:
		semi = array*tempo
		semi = pd.DataFrame(semi)
		semi['s'] = semi[:].sum(axis=1)
		exit = semi['s'].sum(axis=0)
		if maxx == None or exit > maxx:
			maxx = exit
			val = count
		count += 1
	return val

print('\n<<<Pictures in this folder>>>')
for root, dirs, files in os.walk("."):  
    for filename in files:
    	if filename[-3] + filename[-2] + filename[-1] == 'png':
        	print(filename[:-4])

file_name = input('''\nEnter picture name (without ".png")
For Puasson distribution you should have "p" in name of file
For Gauss distribution you should have "g" in name of file\n>''').lower()

type_of_ditr = 'g'
if file_name.find('p') != -1:
	type_of_ditr = 'p'

imframe = Image.open(file_name +'.png')
#get picture in array
npframe = np.array(imframe.getdata())
df = pd.DataFrame(npframe)
width, height = imframe.size
#create a new column for mean of 3pxls
df['value'] = df.mean(axis=1)
df.drop(columns=[0, 1, 2], inplace=True)
#modify data to easier inderstand
less = df[df['value'] < 135].dropna().index
df.loc[less,'value'] = 1
more = df[df['value'] > 100].dropna().index
df.loc[more,'value'] = 0

w = list(range(width))
h = list(range(height))

#create a new dataframe for picture, it looks like a real picture
#but we have black=1 and bright=0
data = pd.DataFrame(index=h, columns=w)
indx_c = 0
stop = 0
for i in range(len(df)):	
	data.iloc[indx_c, stop] = df['value'][i]
	if stop == (width - 1):
		stop = 0
		indx_c += 1
	else:
		stop += 1

#detect digits
numbers = []
rest = 0
rest_column = 0
for row in list(data.index):
	if rest != 0:
		rest -= 1
		continue
	num_detect = 0
	for column in list(data.columns):
		if rest_column != 0:
			rest_column -= 1
			continue
		if column == width - 1 and row == height - 1:
			break
		if data.iloc[row, column] == 1:
			number = []
			num_detect += 1
			while True:
				for x in range(6):
					buff = 0
					for y in range(9):
						try:
							buff += data.iloc[row + y, column - x]
						except: continue
					if buff == 0:
						this_x = column-x+1
						this_y = row
						number.append([this_y, this_x])
						fory = False
						break
				break
			rest_column = 5 - (column - this_x)
			for y in range(9):
				n_row = []
				for x in range(6):
					n_row.append(data.iloc[this_y + y, this_x + x])
				number.append(n_row)
			numbers.append(number)
	if num_detect != 0:
		rest = 9

#list for meta data
metas = []
for j in range(len(numbers)):
	metas.append(numbers[j][0])

#detect full number
numbers_str = ''
for i in range(len(numbers)):
	if i == 0:
		numbers_str += str(detect(numbers[i]))
		continue
	if metas[i][0] == metas[i-1][0]:
		if abs(metas[i][1] - metas[i-1][1]) < 8:
			numbers_str += str(detect(numbers[i]))
		else:
			numbers_str += ' ' + str(detect(numbers[i]))
	else:
		numbers_str += ' ' + str(detect(numbers[i]))

final_num = np.array(numbers_str.split()).astype(int)

#calculation of parameters
num_dict = dict()
for elem in final_num:
	num_dict[elem] = num_dict.get(elem, 0) + 1

N = len(final_num)
x = final_num.mean()
g = (final_num.var()*N/(N-1))**(0.5)

M3 = 0
for num, w in num_dict.items():
	M3 += ((num-x)**(3))*w/N

# print(num_dict, '\n')
print('μ =', x)
print('σ =', g)
print('σμ =', (g**2/N)**(0.5))
print('Г₁ =', M3/g**3)
print('∆σ =', g/(2*(N-1))**(0.5))

if type_of_ditr == 'p':
	bins_number = len(num_dict.keys())
else:
	bins_number = int(len(num_dict.keys())/(g*1/3))

#Pearson test calculating
ft = list()
hi2 = 0
if type_of_ditr == 'p':
	for i in range(len(num_dict.keys())):
		ft = x**list(num_dict.keys())[i]*np.exp(-x)/np.math.factorial(list(num_dict.keys())[i])
		f = list(num_dict.values())[i]
		hi2 += ((f-ft)/f**(0.5))**2
else:
	for i in range(len(num_dict.keys())):
		ft = (1/(np.sqrt(2*np.pi)*g))*np.exp(-0.5*(1/g*(list(num_dict.keys())[i]-x))**2)
		f = list(num_dict.values())[i]
		hi2 += ((f-ft)/f**(0.5))**2
print('ζ =', np.sqrt(hi2))

#our measurements
f1 = plt.figure(1)
n, bins, patches = plt.hist(sorted(final_num), bins = bins_number, align='mid', density=True)
plt.xlabel('n')
plt.title('Практика')
bins = bins.astype(int)

#theory
f2 = plt.figure(2)
if type_of_ditr == 'p':
	y = list()
	for i in range(len(num_dict.keys())):
		y.append(x**i*np.exp(-x)/np.math.factorial(i))
	plt.plot(range(len(num_dict.keys())), y, '-')
else:
	y = ((1 / (np.sqrt(2 * np.pi) * g)) *
     np.exp(-0.5 * (1 / g * (bins - x))**2))
	plt.plot(bins, y, '-')
plt.xlabel('n')
plt.title('Теория')

plt.show()