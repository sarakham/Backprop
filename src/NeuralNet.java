import java.security.acl.LastOwnerException;
import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner	{

	Random rand;
	ArrayList<ArrayList<BPNode>> layers;
	ArrayList<Double> inputNodeValues;			//the values of the attributes
	ArrayList<Integer> targets;				//what the instance should have been classified
	int INPUT_LAYER_INDEX = 0;
	int HIDDENLAYERCOUNT = 2;
	int NODESPERLAYER = 3;
	double LEARNING_RATE = 1.0;
	double MOMENTUM = 1.0;
	
	/*
	 *  Constructor
	 */
	public NeuralNet(Random rand)	{
		this.rand = rand;
		layers = new ArrayList<ArrayList<BPNode>>();
		inputNodeValues = new ArrayList<Double>();
		targets = new ArrayList<Integer>();
	}
	
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
	// create the network of nodes
		int numInstances = features.rows();
		int numInputNodes = features.cols();
		int numOutputNodes = labels.getUniqueValues(0);	//HACK won't work if not all classifications are seen
		
	// get all the instances - input values come from the instance
//		for(int instance = 0; instance < numInstances; instance++)	{	//TODO get all instances
			int instance = 0;
			setInputNodeValues(features);
			setTargets(numOutputNodes, (int)labels.get(instance, 0));

			createNetwork(numInputNodes, numOutputNodes);
			intializeNetworkWeights();
			
			// Pass forward through network, calculating value of each node in each layer
			for(int layerCount = 0; layerCount < layers.size(); layerCount++)	{
				System.out.println("------------------\nLAYER: " + layerCount + "\n------------------");
				passforward(layerCount);
			}
			
			int prediction = computePrediction();
			
			//calculate the error of the output nodes
			computeError(layers.size()-2);
			
			System.out.println("Pause to check answers");
//		}
		
	}

	
	/*
	 * Sets an arrayList with a 1 where the prediction should have
	 * occurred and 0s everywhere else
	 */
	private void setTargets(int numClassifications, int targetValue)	{

		// targetValue is the index into the classifications ArrayList where
		// the correct classification is
		for (int i = 0; i < numClassifications; i++)	{
			if(i == targetValue) {
				targets.add(i, 1);
			}
			else	{
				targets.add(i, 0);
			}
		}
	}
	
	/*
	 * Sets the values of the input nodes to the values of the features
	 * for the instance
	 */
	// TODO make this take in an arbitrary instance and add the values
	private void setInputNodeValues(Matrix features) {
		double[] d = features.row(INPUT_LAYER_INDEX);
		for (int i = 0; i < d.length; i++)	{
			inputNodeValues.add(i, d[i]);
		}
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
		for(int i = 0; i < weights.size(); i++)	{
			double temp = weights.get(i);
			temp = average - temp;
			weights.set(i, temp);
		}
		System.out.println("Average: " + average);
		
		//assign the weights
		for (BPNode node : curLayer)	{
			for(int i = 0; i < prevLayer.size(); i++)	{
				node.weights.add(weights.remove(0));
				node.weightChanges.add(0.0);		//initialize weightChanges 
			}
		}
	}

	
	/*
	 * Picks the highest value output node and returns its index in layers
	 */
	private int computePrediction()	{
		ArrayList<BPNode> outputNodes = layers.get(layers.size()-1);
		double maxValue = -1;
		int maxIndex = -1;
		for(int index = 0; index < outputNodes.size(); index++)	{
			if(outputNodes.get(index).value > maxValue)	{
				maxIndex = index;
				maxValue = outputNodes.get(index).value;
			}
		}
		return maxIndex;
	}
	
	
	/*
	 * Calculate the values of each node in a layer called j by
	 * multiplying the values of the nodes in layer i with 
	 * their weights between nodes in layer i and j
	 * 
	 * Forward propagation
	 */
	private void passforward(int j)	{
		if(j == 0)	{			// set the value of the input nodes 
			ArrayList<BPNode> layer_j = layers.get(j);
			
			for(int n = 0; n < layer_j.size(); n++)	  {
				layer_j.get(n).value = inputNodeValues.get(n);
			}
			System.out.println("layer_j" + layer_j);
		}
		else if(j != 0)	 {		//hidden layers
			ArrayList<BPNode> layer_i = layers.get(j-1);
			ArrayList<BPNode> layer_j = layers.get(j);
	
			//calc value of nodes in hidden layers
			for(int j_node = 0; j_node < layer_j.size(); j_node++)	{
				double nodeValue = 0.0;
				ArrayList<Double> ij_weights = layer_j.get(j_node).weights;

				//sum of weights * i_node values
				for(int i_node = 0; i_node < layer_i.size(); i_node++)	{	
					double curNodeVal = layer_i.get(i_node).value * ij_weights.get(i_node);
					System.out.println("CurNodeValue (product of val and weight): " + curNodeVal);
					layer_j.get(j_node).value = curNodeVal;
					nodeValue += curNodeVal;
				}
//				System.out.println("weights for node j: " + j_count + " " + layer_j.get(j_count));
				System.out.println("\n\tValue before sigmoid: " + nodeValue);
				
				//compute the sigmoid
				nodeValue = sigmoid(nodeValue);
				
				//update the value of the original node
				layers.get(j).get(j_node).value = nodeValue;
				
				System.out.println("\tNode value after sigmoid: " + nodeValue + "\n");
			}
		}
	}

	
	/*
	 * Computes the error of the nodes in the layer
	 */
	private void computeError(int layer_j)	{
		ArrayList<BPNode> layeri = layers.get(layer_j - 1);
		ArrayList<BPNode> layerj = layers.get(layer_j);
		ArrayList<BPNode> layerk = layers.get(layer_j + 1);

		//TODO fix the target value system	
		if(layer_j == layers.size()-2)	{	//output nodes
			
			// (T_k - O_k)*f'(net_k)
			for (int k = 0; k < layerk.size(); k++)	{
				int targetValue = targets.get(k);
				double outputK = layerk.get(k).value;
				double f_prime_net_k = outputK * (1 - outputK);
				
				double errorK = (targetValue - outputK) * f_prime_net_k;
				//TODO weight change for output nodes
				
				for(int j = 0; j < layerj.size(); j++)	{
					double output_j = layerj.get(j).value;
					double weight_change_jk = LEARNING_RATE * output_j * errorK;
					//store the weight change
					layers.get(layer_j + 1).get(k).weightChanges.set(j, weight_change_jk);
				}
			}
		}
		else if (layer_j > 0)	{	// hidden layers
			double layerk_error = 0.0;
			double layerj_error = 0.0;
			double v1 = 0.0;	//weights_jk * errors_k
			double netj = 0.0;	//weights_ij * values_i
			
			// REMOVE - set some test errors
			int count = 0;
			for (BPNode n : layerk)	{
				n.error += .2 * count;
				count += 1.5;
			}
			
			// weights_jk * errors_k
			for (int j = 0; j < layerj.size(); j++)	{		//layer j nodes
				for (int k = 0; k < layerj.size(); k++)	{	//layer k nodes	
					double weight = layerk.get(k).weights.get(j);
					double error = layerk.get(k).error;
					layerk_error += error;
					v1 += weight * error;
				}
			}
			
			//net_j
			
			
			System.out.println("Layer K error: " + layerk_error);
			System.out.println("V1: " + v1);
			System.out.println("Netj: " + netj);
		}
	}
	
	
	/*
	 * Computes the sigmoid function
	 */
	private double sigmoid(double x)	{
		return 1/(1+Math.exp(-x));
	}
	
	
	/*
	 * Computes the derivative of the sigmoid function
	 */
	private double dxSigmoid(double x) 	{
		return Math.exp(x)/Math.pow((Math.exp(x) + 1), 2);
	}
	
	
	/*
	 *  Create the network of nodes
	 */
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

