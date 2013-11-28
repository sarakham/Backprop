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

	// initialize the network
		createNetwork(numInputNodes, numOutputNodes);
		intializeNetworkWeights();
		
	// TODO add outer loop here which is based on stopping criterion
	ArrayList<Double> ave_errors = new ArrayList<Double>();
	// get all the instances - input values come from the instance
		for(int instance = 0; instance < numInstances; instance++)	{
			// TODO generate a random instance number
			setInputNodeValues(features, instance);
			setTargets(numOutputNodes, (int)labels.get(instance, 0));

			// Pass forward through network, calculating value of each node in each layer
			for(int layerCount = 0; layerCount < layers.size(); layerCount++)	{
				System.out.println("------------------\nLAYER: " + layerCount + "\n------------------");
				passforward(layerCount);
			}
			
//			int prediction = computePrediction();
			
			// calculate the error for each layer
			for (int layer = layers.size()-1; layer >= 0; layer--)		{
				computeError(layer);
			}
			
			// update the weights
			for (int layer = layers.size()-1; layer >= 0; layer--)		{
				updateWeights(layer);
			}
			System.out.println("Pause to check answers");
			// get the average error on the output nodes
			double average_output_err = getGeneralError();
			ave_errors.add(instance, average_output_err);
		}
		System.out.println("Pause.");
	}

	/*
	 * Calculates the average error on the ouput nodes 
	 */
	private double getGeneralError()	{
		int outputLayerIndex = layers.size()-1;
		int nodeCount = layers.get(outputLayerIndex).size();
		
		double error_total = 0.0;
		for(int node = 0; node < nodeCount; node++)	{
			error_total += layers.get(outputLayerIndex).get(node).error;
		}
		error_total = error_total/nodeCount;	//average
		return error_total;
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
	private void setInputNodeValues(Matrix features, int instance) {
		double[] d = features.row(instance);
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
	 * Update the weights between layers based upon the stored 
	 * weight changes
	 */
	private void updateWeights(int layer)	{
		int nodeCount = layers.get(layer).size();
		
		for(int node = 0; node < nodeCount; node++)	{
			int weightCount = layers.get(layer).get(node).weights.size();
			for(int w = 0; w < weightCount; w++)	{	//loop through the weights
				double weightChange = layers.get(layer).get(node).weightChanges.get(w); 
				double weight = layers.get(layer).get(node).weights.get(w);
				double updatedWeight = weight + weightChange;
				//update the weight
				layers.get(layer).get(node).weights.set(w, updatedWeight);
			}
		}
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
	 * //TODO add the momentum which will have to be calculated BEFORE 
	 * updating the change in weights 	
	 */
	private void computeError(int layer)	{

		if(layer == layers.size()-1)	{	//output nodes
			ArrayList<BPNode> layerj = layers.get(layer - 1);
			ArrayList<BPNode> layerk = layers.get(layer);
			
			// (T_k - O_k)*f'(net_k)
			for (int k = 0; k < layerk.size(); k++)	{
				int targetValue = targets.get(k);
				double outputK = layerk.get(k).value;
				double f_prime_net_k = outputK * (1 - outputK);
				
				// store error
				double errorK = (targetValue - outputK) * f_prime_net_k;
				layers.get(layer).get(k).error = errorK;
				
				// change in weight 
				for(int j = 0; j < layerj.size(); j++)	{
					double output_j = layerj.get(j).value;
					double weight_change_jk = LEARNING_RATE * output_j * errorK;
					//store the weight change
					layers.get(layer).get(k).weightChanges.set(j, weight_change_jk);
				}
			}
		}
		else if (layer > 0)	{	// hidden layers
			ArrayList<BPNode> layeri = layers.get(layer - 1);
			ArrayList<BPNode> layerj = layers.get(layer);
			ArrayList<BPNode> layerk = layers.get(layer + 1);
			
			// weights_jk * errors_k
			for (int j = 0; j < layerj.size(); j++)	{
				
				double error_layerk = 0.0;	//weights_jk * errors_k
				double outputJ = layerj.get(j).value;
				double f_prime_net_j = outputJ * (1 - outputJ);
				
				// store error
				for (int k = 0; k < layerj.size(); k++)	{		
					double weight_jk = layerk.get(k).weights.get(j);
					double errorK = layerk.get(k).error;
					error_layerk += weight_jk * errorK;
				}
				double error_j = error_layerk * f_prime_net_j;  
				layers.get(layer).get(j).error = error_j;
				
				// weight change
				for (int i = 0; i < layeri.size(); i++)	{
					double output_i = layeri.get(i).value;
					double weight_change_ij = LEARNING_RATE * output_i * error_j;
					//store the weight change
					layers.get(layer).get(j).weightChanges.set(i, weight_change_ij);
				}
			}
		}
	}
	
	
	/*
	 * Computes the sigmoid function
	 */
	private double sigmoid(double x)	{
		return 1/(1+Math.exp(-x));
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

