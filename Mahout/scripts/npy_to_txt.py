from __future__ import print_function

import sys
import os
import numpy as np



if __name__ == "__main__":
	if len(sys.argv) != 3:
        	print("Usage: npy_to_text.py <input .npy> <output .txt>", file=sys.stderr)
        	exit(-1)

	features = np.load(sys.argv[1])
    	id = [i for i in xrange(1,len(features)+1)]
	features_rf = np.insert(features, 0, id, axis=1)
	np.savetxt(sys.argv[2], features, delimiter="\t", fmt=' '.join(['%i'] + ['%1.6f']*2000))
