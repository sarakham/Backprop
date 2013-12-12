import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class NeuralNet extends SupervisedLearner	{

	Random rand;
	ArrayList<ArrayList<BPNode>> layers;
	ArrayList<Double> inputNodeValues;			//the values of the attributes
	ArrayList<Integer> targets;					//what the instance should have been classified
	ArrayList<Integer> numHiddenNodesPerLayer;
	int INPUT_LAYER_INDEX = 0;
	int HIDDENLAYERCOUNT = 1;
//	int NODESPERLAYER = 3;
	double LEARNING_RATE = .5;
	double MOMENTUM = 0.0;
	
	/*
	 *  Constructor
	 */
	public NeuralNet(Random rand)	{
		this.rand = rand;
		layers = new ArrayList<ArrayList<BPNode>>();
		inputNodeValues = new ArrayList<Double>();
		targets = new ArrayList<Integer>();
		numHiddenNodesPerLayer = new ArrayList<Integer>();
		numHiddenNodesPerLayer.add(4);
//		numHiddenNodesPerLayer.add(6);
//		numHiddenNodesPerLayer.add(4);
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception	{
		train(features, labels, null, null);
	}
	
	/*
	 * Train and compute accuracy
	 */
	@Override
	public void train(Matrix features, Matrix labels, Matrix testFeatures, Matrix testLabels) {
		// create the network of nodes
		int numInstances = features.rows();
		int numInputNodes = features.cols();
		int numOutputNodes = labels.getUniqueValues(0);		//HACK won't work if not all classifications are seen
	
		// initialize the network - the if statement is so we don't re-initialize weights when doing cross-fold  
		if (layers.size() == 0)	{
			createNetwork(numInputNodes, numOutputNodes);
			intializeNetworkWeights();
		}
		
		instanceOrder(features);

		ArrayList<Double> trainingErrorsAcrossAllEpochs = new ArrayList<Double>();
		ArrayList<Double> testAccuracyAcrossAllEpochs = new ArrayList<Double>();
		int numEpoch = 0;
		double errorThisEpoch = 1;
//		while((numEpoch < 1000) || (errorThisEpoch > .005))	{
//			numEpoch++;
		for(numEpoch = 0; numEpoch < 30; numEpoch++)	{
			
			ArrayList<Double> thisEpochErrors = new ArrayList<Double>();
			ArrayList<Integer> instanceList = instanceOrder(features);

			for(int instCount = 0; instCount < numInstances; instCount++)	{
				int instance = instanceList.get(instCount);		//get the next instance from the list
				
				// set up instance
				setInputNodeValues(features, instance);
				setTargets(numOutputNodes, (int)labels.get(instance, 0));
	
				// Calculating value of each node in each layer
				for(int layerCount = 0; layerCount < layers.size(); layerCount++)	{
					passforward(layerCount);
				}
				
				//check our prediction
				checkTrainPrediction();
				
				// calculate the error for each layer
				for (int layer = layers.size()-1; layer >= 0; layer--)		{
					computeError(layer);
				}
				
				// update the weights
				for (int layer = layers.size()-1; layer >= 0; layer--)		{
					updateWeights(layer);
				}
				
				// compute training error for epoch - get the error on the output nodes
				double instance_error = calcOutputNodeError();
				thisEpochErrors.add(instance_error);
			}
			
			// print epoch number
			if (numEpoch % 10 == 0)	{
				System.out.println("Epoch " + numEpoch);
			}
			
			if(numEpoch > 3500)	{
				break;
			}
			
			// calculate training error for the epoch
			errorThisEpoch = calcAverageError(thisEpochErrors);
			trainingErrorsAcrossAllEpochs.add(errorThisEpoch);
			
			// calculate the test accuracy for the epoch
			if(testFeatures != null && testLabels != null)	{
				try {
					double accuracy = super.measureAccuracy(testFeatures, testLabels, null);
//					System.out.println("Epoch: " + numEpoch + " accuracy: " + accuracy);
					testAccuracyAcrossAllEpochs.add(accuracy);
					
				}
				catch (Exception e) {
					System.out.println("An exception was handled while computing the test accuracy");
					e.printStackTrace();
				}
			}
		}
		
		System.out.println("Finished Training with " + numEpoch);
		writeArrayListToFile(trainingErrorsAcrossAllEpochs, "1_learn rate_allEpochErrors");
		writeArrayListToFile(testAccuracyAcrossAllEpochs, "1_learn_rate_allEpochTestAccuracy");
	}
	
	
	/*
	 * Returns a random order of instances
	 */
	private ArrayList<Integer> instanceOrder(Matrix features)	{
		
		ArrayList<Integer> instanceOrder = new ArrayList<Integer>();
		for (int i = 0; i < features.rows(); i++)	{
			instanceOrder.add(i);
		}
		
		Collections.shuffle(instanceOrder, new Random());
		return instanceOrder;
	}
	
	
	/*
	 * Returns the sum of errors across the output nodes 
	 */
	private double calcOutputNodeError()	{
		int outputLayerIndex = layers.size()-1;
		int nodeCount = layers.get(outputLayerIndex).size();
		
		double error_total = 0.0;
		for(int node = 0; node < nodeCount; node++)	{
			error_total += Math.abs(layers.get(outputLayerIndex).get(node).error);
		}
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
//		System.out.println("Average: " + average);
		
		//assign the weights
		for (BPNode node : curLayer)	{
			for(int i = 0; i < prevLayer.size(); i++)	{
				node.weights.add(weights.remove(0));
				node.weightChanges.add(0.0);		//initialize weightChanges 
			}
		}
	}

	
	/*
	 * Determines if we made the correct prediction
	 */
	private void checkTrainPrediction()	{
		int predictionIndex = computePrediction();
		
		//find the 1 in targets
		int classificationIndex = -1;
		for(int index = 0; index < targets.size(); index++)	{
			if(targets.get(index) == 1)	{
				classificationIndex = index;
				break;
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
					layer_j.get(j_node).value = curNodeVal;
					nodeValue += curNodeVal;
				}
				
				//compute the sigmoid
				nodeValue = sigmoid(nodeValue);
				
				//update the value of the original node
				layers.get(j).get(j_node).value = nodeValue;
			}
		}
	}

	
	/*
	 * Computes the error of the nodes in the layer
	 * //TODO add the momentum which will have to be calculated BEFORE 
	 * updating the change in weights 	
	 */
	private void computeError(int layer)	{

		if(layer == layers.size()-1)	{	// output nodes
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
			
			// weights_jk * errors_k - j is a node in layerj
			for (int j = 0; j < layerj.size(); j++)	{
				
				double error_layerk = 0.0;	//weights_jk * errors_k
				double outputJ = layerj.get(j).value;
				double f_prime_net_j = outputJ * (1 - outputJ);
				
				// store error
				for (int k = 0; k < layerk.size(); k++)	{		
					double weight_jk = layerk.get(k).weights.get(j);
					double errorK = layerk.get(k).error;
					error_layerk += weight_jk * errorK;
				}
				double error_j = error_layerk * f_prime_net_j;  
				layers.get(layer).get(j).error = error_j;
				
				// weight change
				for (int i = 0; i < layeri.size(); i++)	{
					double output_i = layeri.get(i).value;
					double momentum = 0.0;
					//calculate momentum term
					if(layers.get(layer).get(j).weightChanges.get(i) != null)	{
						double previous = layers.get(layer).get(j).weightChanges.get(i);
						momentum = MOMENTUM * previous;
					}
					double weight_change_ij = LEARNING_RATE * output_i * error_j + momentum;
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
	 * Calculates the average error of an epoch
	 */
	private double calcAverageError(ArrayList<Double> errors)	{
		double aveError = 0.0;
		for(int i = 0; i < errors.size(); i++)	{
			aveError += errors.get(i);
		}
		aveError = aveError/errors.size();
		return aveError;
	}
	
	
	/*
	 * Writes an ArrayList to a .txt file
	 */
	public void writeArrayListToFile(ArrayList<Double> list, String name) {
		String filename = name + ".txt";
		
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename));
			
			for(int i = 0; i < list.size(); i++)	{
				out.println(list.get(i));
			}

			out.close();
		} catch (IOException e) {
			System.out.println("Failed to make PrintWriter");
			e.printStackTrace();
		}
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
		for(int i = 1; i <= numHiddenNodesPerLayer.size(); i++)	{
			layers.add(i, new ArrayList<BPNode>());
			for (int j = 0; j < numHiddenNodesPerLayer.get(i-1); j++)		{
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
		
//		System.out.println("Prediction");
		
		// set input node values
		for (int i = 0; i < features.length; i++)	{
			inputNodeValues.add(i, features[i]);
		}
		
		// Calculating value of each node in each layer
		for (int numLayer = 0; numLayer < layers.size(); numLayer++)	{
			passforward(numLayer);
		}

		// calculate the error for each layer
		for (int layer = layers.size()-1; layer >= 0; layer--)		{
			computeError(layer);
		}
		
		//check our prediction - index of output node with largest value
		ArrayList<BPNode> outputNodes = layers.get(layers.size()-1);
		int prediction = 0;
		double max = -1;
		for (int index = 0; index < outputNodes.size(); index++)	{
			if(outputNodes.get(index).value > max)	{
				max = outputNodes.get(index).value;
				prediction = index;
			}
		}
		
		//assign the label
		labels[0] = prediction;
	}
}