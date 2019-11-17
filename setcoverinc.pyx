# distutils: language=c++
import numpy as np
import math
import copy
import anchor_tabular
from libcpp.vector cimport vector
from libcpp.set cimport set

cpdef low_deg_partial(int X, int xstar, double miu, b_sets, r_sets,int k,int m,sets_feature):
	cdef vector[set[int]] r_Sx
	cdef vector[set[int]] b_Sx
	#cdef vector[int] Sx_idx
	Sx_idx=[]
	for i in range(len(b_sets)):
		if (len(r_sets[i]) <= X):
			r_Sx.push_back(r_sets[i])
			b_Sx.push_back(b_sets[i])
			#Sx_idx.push_back(i)
			Sx_idx.append(i)

	#print(Sx_idx)

	cdef int n_blue = m
	bitmap = np.zeros(n_blue)
	cdef int blue_counter = 0
	for s in b_Sx:
		for element in s:
			if (bitmap[element] == 0):
				bitmap[element] = 1
				blue_counter = blue_counter + 1
	if (blue_counter / n_blue < miu):
		return 1, None

	cdef int d = len(r_Sx)
	cdef int beta = n_blue
	cdef double Y = math.sqrt(d / harmonic_n(beta))
	print('Y=')
	print(Y)

	bitmap = np.zeros(n_blue)
	for s in r_Sx:
		for element in s:
			bitmap[element] += 1

	# r_Sxy=list(r_Sx)
	# b_Sxy=list(b_Sx)

	#r_Sxy = copy.deepcopy(r_Sx)
	r_Sxy=vector[set[int]](r_Sx)
	#b_Sxy = copy.deepcopy(b_Sx)
	b_Sxy=vector[set[int]](b_Sx)

	for i in range(n_blue):
		if (bitmap[i] > Y):
			for s in r_Sxy:
				if i in s:
					s.remove(i)
					
	#print r_Sxy

	cdef double ratio
	cdef set[int] cover
	cover,ratio = greedy_partial_RB(r_Sxy, b_Sxy, miu, xstar,m)
	print('ratio')
	print(ratio)
	# print(cover)
	# for i in cover:
	#     print Sx_idx[i]
	#     print self.b_sets[Sx_idx[i]]

	bitmap_r = np.zeros(n_blue)
	bitmap_b = np.zeros(n_blue)
	cdef int n_covered_red = 0
	cdef int n_covered_blue = 0
	kdnf = []
	cdef int idx
	for idx in cover:
		for element in r_Sx[idx]:
			if (bitmap_r[element] == 0):
				n_covered_red += 1
				bitmap_r[element] = 1
		for element in b_Sx[idx]:
			if (bitmap_b[element] == 0):
				n_covered_blue += 1
				bitmap_b[element] = 1
		kdnf.append(sets_feature[Sx_idx[idx]])

	cdef double err = n_covered_red * 1.0 / n_covered_blue
	print('err')
	print(err)
	print('kdnf')
	print(kdnf)
	#print(anchor_tabular.compute_precision_coverage(kdnf))

	return err, kdnf

cdef double harmonic_n(int x):
	cdef double sum = 0
	cdef int i
	for i in range(1, x + 1):
		sum = sum + 1. / i
	return sum

cdef greedy_partial_RB(vector[set[int]] & S_r, vector[set[int]] & S_b, double miu, int xstar,m):
	cdef vector[int] weights
	for s in S_r:
		weights.push_back(s.size())
	#print(weights)
	return partial_greedy(S_b, weights, miu, xstar,m)

cdef partial_greedy( vector[set[int]] & sets, vector[int] & weights, double miu, int xstar,m):
	#for k in range(len(sets)):
	#	print len(sets[k])
	#best_cover = []
	cdef double best_ratio = 999999999
	cdef int i
	cdef int j
	cdef int total_weight
	cdef int total_size
	cdef double low_ratio
	cdef int choice
	cdef set[int] cover
	for i, s in enumerate(sets):
		if (xstar in s):
			#print i
			#print('i')
			#print i
			cover.clear()
			cover.insert(i)
			total_weight = weights[i]
			total_size = len(s)
			ratios = np.zeros(len(sets))
			ratios[i] = -1
			#new_sets = copy.deepcopy(sets)
			new_sets=vector[set[int]](sets)
			for j in range(len(new_sets)):
				if (cover.find(j)==cover.end()):
					new_sets[j] = new_sets[j].difference(s)
					if (len(new_sets[j]) > 0):
						ratios[j] = weights[j]*1.0 / len(new_sets[j])
					else:
						ratios[j] = -1
			
			#for k in range(len(sets)):
			#	print len(sets[k])

			while miu * m > total_size:
				low_ratio = 99999999
				choice = -1
				for j in range(len(sets)):
					if (cover.find(j)==cover.end() and len(new_sets[j]) > 0 and ratios[j] < low_ratio):
						low_ratio = ratios[j]
						choice = j
				if (choice != -1):
					cover.insert(choice)
					total_weight += weights[choice]
					total_size += len(new_sets[choice])
					for j in range(len(new_sets)):
						if (cover.find(j)==cover.end() and len(new_sets[j]) > 0):
							new_sets[j] = new_sets[j].difference(new_sets[choice])
							if (len(new_sets[j]) > 0):
								ratios[j] = weights[j]*1.0 / len(new_sets[j])
							else:
								ratios[j] = -1
				else:
					print "exception"
					print sets
					print i
					print cover
					raise Exception("no choice")

			while (1):
				low_ratio = 99999999
				choice = -1
				for j in range(len(sets)):
					if (cover.find(j)==cover.end() and len(new_sets[j]) > 0 and ratios[j] < low_ratio):
						low_ratio = ratios[j]
						choice = j
				if (choice != -1 and low_ratio < total_weight * 1.0 / total_size):
					cover.insert(choice)
					total_weight += weights[choice]
					total_size += len(new_sets[choice])
					for j in range(len(new_sets)):
						if (cover.find(j)==cover.end() and len(new_sets[j]) > 0):
							new_sets[j] = new_sets[j].difference(new_sets[choice])
							if (len(new_sets[j]) > 0):
								ratios[j] = weights[j]*1.0 / len(new_sets[j])
							else:
								ratios[j] = -1
				else:
					break

			if (total_weight * 1.0 / total_size < best_ratio):
				best_cover = cover
				best_ratio=total_weight * 1.0 / total_size

	return best_cover,best_ratio

