#!/usr/bin/python
# --coding: utf-8 --  

# Gabriel Baptista nusp: 8941300

import sys
from math import log
from util import Counter

def gini( labels ) :
	freq = {}

	for l in set(labels):
		freq[l] = labels.count(l)
	f = Counter(freq) # frequencia

	if len(f) == 2:
		p1 = float(f[0]) / (float(f[0]) + float(f[1])) # probabilidade 1
		p2 = float(f[1]) / (float(f[0]) + float(f[1])) # probabilidade 2
		return (1 - (p1 * p1 + p2 * p2))
	else:
		return 0
		 			

def error( labels ) :
	freq = {}

	for l in set(labels):
		freq[l] = labels.count(l)
	f = Counter(freq) # frequencia

	if len(f) == 2:
		p1 = float(f[0]) / (float(f[0]) + float(f[1])) 
		p2 = float(f[1]) / (float(f[0]) + float(f[1])) 
		return (1 - max(p1, p2))
	else:
		return 0 

def entropy( labels ) :
	freq = {}

	for l in set(labels):
		freq[l] = labels.count(l)
	f = Counter(freq) # frequencia

	if len(f) == 2:
		p1 = float(f[0]) / (float(f[0]) + float(f[1])) 
		p2 = float(f[1]) / (float(f[0]) + float(f[1])) 
		return (-(p1 * log(p1) + p2 * log(p2))) 
	else:
		return 0
	
