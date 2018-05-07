package bookrecommender;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
public class RecommendationMain {
	
	public static void main(String[] args) throws NumberFormatException, IOException, TasteException {
		// TODO Auto-generated method stub
		DataModel model = null;
		try {
			model = new FileDataModel(new File (args[0]));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		boolean content_based=false;
		if(args.length==2)
			content_based=true;
		ItemBasedRecommender recommender_item =null;
		if(content_based){
		
		
		//TODO: Mem optimization to load 1.2 GB 10k*10k Book Feature Similarity Matrix for Item Based Rec
		List<List<Float>> features = new ArrayList<List<Float>>();
		BufferedReader br = new BufferedReader(new FileReader(new File(args[1]))); // reads item-item content similarity file 
		 
		  String st;
		  StringTokenizer st1 = null;
		  List<Float> temp=null;
		  while ((st = br.readLine()) != null) {
			temp=new ArrayList<Float>();  		
	        st1 = new StringTokenizer(st, " ");
		    st1.nextToken(); //skip id
	        while (st1.hasMoreTokens()) {
	            temp.add(Float.parseFloat(st1.nextToken()));
	        }
	        features.add(temp);
		  }
		  ItemSimilarity book_similarity = new BookSimilarity(features);
		  ItemSimilarity item_similarity = new GenericItemSimilarity(book_similarity,model, 100);
		  recommender_item = new GenericItemBasedRecommender(model, book_similarity);
		  
		}
		
		/*
		 //Item Based CF 
		 ItemSimilarity itemSimilarity = new LogLikelihoodSimilarity(dataModel);
		//Create an Item Based Recommender
		ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel, itemSimilarity);
		*/
		
		// User Based CF  
		UserSimilarity similarity =  null;
		
		similarity = new PearsonCorrelationSimilarity(model);
		
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
		
		UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
		
		// 
		int book_id_to_rec_for = 2;
		int num_recs = 10;
		List<RecommendedItem> recommendations=null;
		try {
			if(!content_based)	
				recommendations = recommender.recommend(book_id_to_rec_for, num_recs);
			else recommendations = recommender_item.recommend(book_id_to_rec_for, num_recs);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (RecommendedItem recommendation : recommendations) {
		  System.out.println(recommendation);
		}
		
		//Evaluations
		RecommenderEvaluator evaluator_map = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderEvaluator evaluator_rmse = new RMSRecommenderEvaluator();
		RecommenderBuilder builder = new MyRecommenderBuilder();
		double result_map=0, result_rmse=0;
		try {
			result_map = evaluator_map.evaluate(builder, null, model, 0.9, 1.0);
			result_rmse = evaluator_rmse.evaluate(builder, null, model, 0.9, 1.0);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("RMSE: "+result_rmse);
		System.out.println("MAP: "+result_map);
		
	}

}
