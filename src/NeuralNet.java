import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner	{

	Random rand;
	ArrayList<ArrayList<BPNode>> layers;
	int HIDDENLAYERCOUNT = 2;
	int NODESPERLAYER = 3;
	double LEARNINGRATE = 3.0;
	
	// Constructor
	public NeuralNet(Random rand)	{
		this.rand = rand;
		layers = new ArrayList<ArrayList<BPNode>>();
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		// create the network of nodes
		int numInputNodes = features.cols();
		int numOutputNodes = labels.getUniqueValues(0);
		
		createNetwork(numInputNodes, numOutputNodes);
		intializeNetworkWeights();
		setLayerValues(1);
		
	}
	
	/*
	 * Sets the weights for all connections in this fully-connected network
	 * 	Input nodes receive no weight
	 */
	private void intializeNetworkWeights() {

		//layers
		for(int i = 0; i < layers.size(); i++)		{
			if(i != 0)	
				genLayerWeights(i);
		}
	}
	
	/*
	 * Generates the weights for a layer
	 */
	private void genLayerWeights(int layerIndex)	{
		
		ArrayList<BPNode> prevLayer = layers.get(layerIndex-1);
		ArrayList<BPNode> curLayer = layers.get(layerIndex);
		int weightsNeeded = prevLayer.size() * curLayer.size();
		
		//generate weights within the range
		ArrayList<Double> weights = new ArrayList<Double>();
		double average = 0.0;
		double rangeMin = -.1;
		double rangeMax = .1;
		
		for(int i = 0; i < weightsNeeded; i++)	{
			double tempweight = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
			average += tempweight;
			weights.add(tempweight);
		}
		average /= weights.size();
		
		//adjust to get a mean of 0
		for (Double w : weights)	{
			w -= average;
		}
//		System.out.println("Average: " + average);
		
		//assign the weights
		for (BPNode node : curLayer)	{
			for(int i = 0; i < prevLayer.size(); i++)	{
				node.weights.add(weights.remove(0));
			}
		}
	}

	/*
	 * Calculate the values of each node in a layer called j
	 * This multiplies the values of the nodes in layer i with 
	 * their weights leading to the node in layer j
	 */
	private void setLayerValues(int j)	{
		
		if(j != 0)	{
			ArrayList<BPNode> layer_i = layers.get(j-1);
			ArrayList<BPNode> layer_j = layers.get(j);
			
			double totalValue = 0.0;
			for(int j_count = 0; j_count < layer_j.size(); j_count++)	{
				ArrayList<Double> node_weights = layer_j.get(j_count).weights;
				//calculate sum of weights * values from layer i
				for(int i_count = 0; i_count < layer_i.size(); i_count++)	{
					double curNodeVal = layer_i.get(i_count).value * node_weights.get(i_count);
					layer_j.get(j_count).value = curNodeVal;
					totalValue += curNodeVal;
				}
//				System.out.println("weights for node j: " + j_count + " " + layer_j.get(j_count));
				System.out.println("\n\tValue: " + totalValue);
			}
			System.out.println("Total value for layer: " + totalValue);
		}
	}
	
	
	// Create the network of nodes
	private void createNetwork(int numInputNodes, int numOutputNodes)	{
		
		// Input Nodes - as many as there are attributes
		layers.add(0, new ArrayList<BPNode>());
		for(int i = 0; i < numInputNodes; i++)	{
			layers.get(0).add(i, new BPNode("Input"));
			layers.get(0).get(i).value = 1;
		}
		
		// Hidden Nodes - number chosen
		for(int i = 1; i <= HIDDENLAYERCOUNT; i++)	{
			layers.add(i, new ArrayList<BPNode>());
			for (int j = 0; j < NODESPERLAYER; j++)		{
				layers.get(i).add(j, new BPNode("Hidden"));
			}
		}
		
		// Output Nodes - as many as there are classifications
		int lastCol = layers.size();
		layers.add(lastCol, new ArrayList<BPNode>());
		for(int i = 0; i < numOutputNodes; i++)	{
			layers.get(lastCol).add(i, new BPNode("Output"));
		}
	}
	
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	
	
	
	//------------------------------------- dead code ----------------------------------------------
//	/*
//	 * Sets the weights for all connections in this fully-connected network
//	 * 	Input nodes receive no weight
//	 */
//	private void intializeNetworkWeights() {
//
//		for(int i = 0; i < layers.size(); i++)	//layers	
//		{	
//			for(int j = 0; j < layers.get(i).size(); j++)	//layer's nodes 	
//			{	
////				System.out.println("i: " + i + " j: " + j);
//				if(i != 0)	// if hidden or output node, one weight for each	
//				{ 	
//					int prevLayerSize = layers.get(i-1).size(); 
//					for(int k = 0; k < prevLayerSize; k++)	//each node in previous layer	
//					{	
//						double weight = rand.nextGaussian();
//						layers.get(i).get(j).inputWeights.add(weight);
////						System.out.println("weight: " + weight);
//					}
//				} 
//			}
//		}
//	}

//	/*
//	 * Sets the weights for all connections in this fully-connected network
//	 * 	Input nodes receive no weight
//	 */
//	private void intializeNetworkWeights() {
//
//		for(int i = 0; i < layers.size(); i++)	
//		{	//layers
//			for(int j = 0; j < layers.get(i).size(); j++) 	
//			{	//layer's nodes
////				System.out.println("i: " + i + " j: " + j);
//				if(i != 0)	
//				{ 	// if hidden or output node, one weight for each
//					int prevLayerSize = layers.get(i-1).size(); 
//					for(int k = 0; k < prevLayerSize; k++)	
//					{	//each node in previous layer
//						double weight = rand.nextGaussian();
//						layers.get(i).get(j).inputWeights.add(weight);
////						System.out.println("weight: " + weight);
//					}
//				} 
//			}
//		}
//	}
	
	/*
	 * TEST - create a distribution with a mean of 0
	 */
	public void createMean()	{
		ArrayList<Double> weights = new ArrayList<Double>();
		
		for(int i = 0; i < 6; i++)	{
			weights.add(rand.nextGaussian());
		}
		
		double sum = 0.0;
		for(Double d : weights)
			sum += d;
		double mean = sum/weights.size();
		
		System.out.println("weights: " + weights);
		System.out.println("mean 1: " + mean);
		
		for(Double d : weights)
			d = d - mean;
		
		sum = 0.0;
		for(Double d : weights)
			sum += d;
		mean = sum/weights.size();

		System.out.println("weights: " + weights);
		System.out.println("mean 1: " + mean);
		System.out.println("Done");
	}
	
	
}
