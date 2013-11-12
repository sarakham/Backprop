import java.math.RoundingMode;
import java.text.DecimalFormat;
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
		//forwardPropagate();
		//doSomething(1);
	}

	
	/*
	 * Sets the weights for all connections in this fully-connected network
	 * 	Input nodes receive no weight
	 */
	private void intializeNetworkWeights() {

		for(int i = 0; i < layers.size(); i++)	//layers	
		{	
			if(i != 0)	{
				genLayerWeights(i);
			}
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
		}
	}
	
	
	/*
	 * Generates the weights for a layer
	 */
	private void genLayerWeights(int layerIndex)	{
		
		ArrayList<BPNode> prevLayer = layers.get(layerIndex-1);
		ArrayList<BPNode> curLayer = layers.get(layerIndex);
		int weightsNeeded = prevLayer.size() * curLayer.size();
		
		//generate weights
		ArrayList<Double> weights = new ArrayList<Double>();
		for(int i = 0; i < weightsNeeded; i++)	{
			weights.add(rand.nextDouble());
		}
		
		System.out.println("weights: " + weights);
		
		//find average
		double average = 0.0;
		for (Double w : weights)	{
			average += w;
		}
		average /= weights.size();
		
		System.out.println("Average: " + average);
		
		//adjust to get a mean of 0
		for (Double w : weights)	{
			w -= average;
		}
		
		System.out.println("weights: " + weights);
		//for reach node in curLayer, generate preLayer.count weights.
		//average all the weights in this list and subtract it from each of the weights
	}


	/*
	 * Calculates values of the nodes during forward propagation
	 */
	private void forwardPropagate()	{
		
		for(int i = 1; i < layers.size(); i++)	
		{	//layers
			for(int j = 0; j < layers.get(i).size(); j++) 	
			{	//layer's nodes
				System.out.println("\ni: " + i + " j: " + j);
				
				double tot = 0.0;
				ArrayList<BPNode> prev_layer = layers.get(i-1);
				System.out.println("prev_layer: " + prev_layer.toString());
				
//					for(int k = 0; k < prev_layer.size(); k++)
//					{	//each node connected in previous layer
//						//add products of weights * values 
//						assert(prev_layer.size() == layers.get(i).get(j).inputWeights.size());
//						System.out.println("prev weight: " + k + " is: " + prev_layer.get(k));
//					}
			}
		}
	}
	
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
	 *  Given layers i and j, this method multiplies the values of the previous 
	 *  nodes with the weight for them and  
	 */
	private void doSomething(int i)	{
		
		ArrayList<BPNode> layer_I = layers.get(i);
		int nodeCount = layer_I.size();
		double value = 0.0;
		for(int node = 0; node < nodeCount; node++)	{
			double prevValue = layer_I.get(node).value;
			double weight = layer_I.get(node).inputWeights.get(i-1);
			System.out.println("node: " + node + " prevValue: " + prevValue + " weight: " + weight);
			value += prevValue * weight;
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
}
