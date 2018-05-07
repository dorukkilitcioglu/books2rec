package bookrecommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

public class BookSimilarity implements ItemSimilarity {
	//TODO Assumes use of contiguous book ids (1-N) . Clean up for all possibilities use HashMap 
	List<List<Float>> item_similarities=null;
	public BookSimilarity(List<List<Float>> item_s) {
		item_similarities = item_s;
		System.out.println(item_similarities.size());
		for(int i=0; i<item_similarities.size(); i++) {
			System.out.println(item_similarities.get(i).size());
		}
	}
	@Override
	public double[] itemSimilarities(long arg0, long[] arg1) throws TasteException {
		double[] result = new double[arg1.length];
		
		for(int i=0; i<arg1.length; i++) {
			result[i] = itemSimilarity(arg0, arg1[i]);
		}
		return result;
	}

	@Override
	public double itemSimilarity(long arg0, long arg1) {
		if(arg0 == arg1) return 1.0;
		double res = 0;
		try {
			res=item_similarities.get((int)arg0-1).get((int)arg1-1);
		} catch(Exception e) {
			e.printStackTrace();
			System.out.println("arg0:"+(arg0-1)+" arg1:"+(arg1-1));
		}
		return res;
	}

	@Override
	public void refresh(Collection<Refreshable> arg0) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public long[] allSimilarItemIDs(long arg0) throws TasteException {
		// TODO set creteria for similar items 
		long[] res = new long[100];
		//Collections.sort(item_similarities.get(((int)arg0));
		for(int i=0; i<100; i++) {
			//res[i] = sorted.get(i).getKey();
		}
		return res;
	}

}
