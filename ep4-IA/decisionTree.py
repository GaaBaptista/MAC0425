#!/usr/bin/python
# --coding: utf-8 -- 

# Gabriel Baptista nusp: 8941300

from copy import deepcopy as copy
from util import Counter
import classificationMethod


class DecisionNode :
	def __init__( self , column = -1 , value = None , label = None , leftchild = None , rightchild = None, depth = None ) :
		self.column = column
		self.depth = depth
		self.value = value
		self.label = label
		self.leftchild = leftchild
		self.rightchild = rightchild

	def __str__( self ) :
		"""
		Return a string representing the structure of DecisionNode
		This function will be implicitly called when printed a DecisionNode object using print function
		"""
		return 'Node'

class DecisionTreeClassifier( classificationMethod.ClassificationMethod ) :
	def __init__( self , legalLabels ) :
		self.guess = None
		self.type = "decisiontree"

	def train( self , data , labels , args ) :
		"""
		Learn the tree model
		"""
		self.maxdepth = int( args[ 'maxdepth' ] )
		self.metric = args[ 'metric' ]
		self.numrows = len( data )
		self.numcolumns = len( data[ 0 ] )
		self.labels = labels		
		self.tree = self.buildTree( data , labels )

	def buildTree( self , data , labels , depth = 0 ) :
		""" Recursive function to learn the tree model """
		if not(self.isLeaf(data, labels, depth)):
			split = self.bestSplit(data, labels)

			dataLeft, labelsLeft, dataRight, labelsRight = self.divideSet(data, labels, split)
			
			left = self.buildTree(dataLeft, labelsLeft, depth + 1)
			right = self.buildTree(dataRight, labelsRight, depth + 1)
			
			tree = DecisionNode(leftchild = left, rightchild = right, column = split)
			return tree
		else:
			if labels == []:
				labels = self.labels
			
			leaf = DecisionNode(label = self.getMostFrequentLabel(labels))
			return leaf 
			

	def isLeaf( self , data , labels , depth ):
		""" Verify stop conditions (whether to split is necessary) """
		freq = {}

		for l in set(labels):
			freq[l] = labels.count(l)
		f = Counter(freq) # frequencia
		
		if depth >= self.maxdepth or len(f) == 1 or not f:
			return True

		elif data != []:
			node = []
			for i in data:
				node.append(data[0] == i)
			if all(node) == True:
				return True
		
		else:
			return False  


	def getMostFrequentLabel( self , labels ):
		freq = {}

		for l in set(labels):
			freq[l] = labels.count(l)
		
		return Counter(freq).argMax()

	def bestSplit( self , data , labels ):
		""" Get the best variable to split the dataset using the metric function """
		info = self.metric(labels)
		split = None
		bestGain = -float('inf')

		for feat in range(self.numcolumns):
			i, labels0, j, labels1 = self.divideSet(data, labels, feat)
			remain0 = remain1 = 0
			if labels0 != []:
				remain0 = float(len(labels0)) / len(labels) * self.metric(labels0)
			if labels1 != []:
				remain1 = float(len(labels1)) / len(labels) * self.metric(labels1)
			
			gain = info - (remain0 + remain1)
			if gain > bestGain:
				bestGain = gain
				split = feat
		
		return split

	def divideSet( self , data , label , variable ) :
		"""
		Given a variable, split the data set and labels in two sets.
		One data set with instances having variable as 0 and the
		other with instances having variable as 1
		"""

		dataLeft = [] 
		labelsLeft = []
		dataRight = []
		labelsRight = []

		for i, test in enumerate(data):
			if test[variable] != 0:
				dataRight.append(test)
				labelsRight.append(label[i])
			else:
				dataLeft.append(test)
				labelsLeft.append(label[i])

		return dataLeft,labelsLeft,dataRight,labelsRight

	def classify( self , testData ) :
		"""
		Classify all test data using the learned tree model
		"""
		labels = []

		for test in testData:
			tree = self.tree
			while (1):
				if tree.column != -1:
					if test[tree.column] != 0:
						tree = tree.rightchild
					else:
						tree = tree.leftchild
				else:
					labels.append(tree.label)
					break
					
		return labels

