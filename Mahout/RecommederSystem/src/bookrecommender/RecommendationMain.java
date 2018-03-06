package bookrecommender;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
public class RecommendationMain {
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DataModel model = null;
		try {
			model = new FileDataModel(new File ("../../project/data/goodbooks-10k-master/ratings_small.csv"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//UserSimilarity similarity =  null;
		ItemSimilarity item_similarity = null;
		//similarity = new PearsonCorrelationSimilarity(model);
		item_similarity = new LogLikelihoodSimilarity(model);
		
		//UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
		
		//UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
		
		ItemBasedRecommender recommender_item = new GenericItemBasedRecommender(model, item_similarity);
		
		List<RecommendedItem> recommendations=null;
		try {
			//recommendations = recommender.recommend(2, 3);
			recommendations = recommender_item.recommend(2, 10);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (RecommendedItem recommendation : recommendations) {
		  System.out.println(recommendation);
		}
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new MyRecommenderBuilder();
		double result=0;
		try {
			result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(result);
		
	}

}
