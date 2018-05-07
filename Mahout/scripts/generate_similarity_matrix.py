from __future__ import print_function

import sys
import os
import numpy as np
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg import Vectors


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: spark-submit generate_similarity_matrix.py <input path to hdfs file> <hdfs output path>", file=sys.stderr)
		exit(-1)
	#convert and process raw input to (bookid, [features])
	def processFeatures(raw) :
		features_str = raw.split();
		book_id = int(features_str[0])
		features = []
		for i in range(1, len(features_str)):
			features.append(float(features_str[i]))
		return (book_id, features) 


	sc = SparkContext(appName="BookRecSystem")
	spark = SQLContext(sc)
	featureRdd = sc.textFile(sys.argv[1])
	featureRdd = featureRdd.map(processFeatures)
	labels = featureRdd.map(lambda x: x[0]) #label_rdd
	fvecs =  featureRdd.map(lambda x: Vectors.dense(x[1])) #feature_rdd
	data = labels.zip(fvecs)
	mat = IndexedRowMatrix(data).toBlockMatrix() #convert to block-matrix for pairwise cosine similarity
	dot = mat.multiply(mat.transpose()).toIndexedRowMatrix().rows.map(lambda x : (x.index, x.vector.toArray())).sortByKey().map(lambda x: str(x[0])+' '.join(map(str, x[1]))) #pairwise_cosine_similarity to rdd
	dot.saveAsTextFile(sys.argv[2]) #save output
	sc.stop()
