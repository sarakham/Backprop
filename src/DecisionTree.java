import java.text.AttributedCharacterIterator.Attribute;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class DecisionTree extends SupervisedLearner {

	int NUM_ATTRIBUTES = 0;
	String SPLITTING_CRITERION = "continuous";
	Matrix features_original;
	Matrix labels_original;

	/*
	 * Constructor
	 */
	public DecisionTree()	{
		
	}
	
	/* 
	 * 	Train
	 */
	public void train(Matrix features, Matrix labels) throws Exception {
		
		NUM_ATTRIBUTES = features.cols();
		
		// list of attributes to split on
		ArrayList<Integer> attributesToSplitOn = new ArrayList<Integer>();
		for(int attribute = 0; attribute < NUM_ATTRIBUTES; attribute++)	{
			attributesToSplitOn.add(attribute);
		}
				
		// build tree
		DTNode root = induceTree(features, labels, attributesToSplitOn);
		System.out.println(root);
	}

	
	/*
	 *  Create the tree
	 */
	public DTNode induceTree(Matrix features, Matrix labels, ArrayList<Integer> attributesToSplitOn)	{
		System.out.println("--------------------\n Current Root Node"); printMatrices(features, labels);
		System.out.println("\nattributeList: " + attributesToSplitOn);
		
		// Make node of the current tree
		DTNode root = new DTNode(features, labels);
		
		double unanimousClassification = checkIfAllElementsShareClass(features, labels);
		
		if(unanimousClassification >= 0)	{		// all instances have the same classification - return leaf node labeled with that class
			root.classification = unanimousClassification;
			System.out.println("\nLeaf node classification: " + root.classification);
			System.out.println("-------------------------------\n");
		}
		else if(attributesToSplitOn.size() == 0)	{	//return a leaf node labeled as the majority class
			root.classification = labels.mostCommonValue(0);
			System.out.println("\nLeaf node classification is majority class: " + root.classification);
			System.out.println("-------------------------------\n");
		}
		else	{
			int selectedAttribute = Entropy.getHighestInformationGain(features, labels, attributesToSplitOn);
			if(selectedAttribute == 1.0)	{
				System.out.println("Pause");
			}
			root.attribute = selectedAttribute;
			System.out.println("selectedAttributeColumn: " + selectedAttribute + " (" + features.attrName(selectedAttribute) + ")  unanimousClassification: " + unanimousClassification);
			removeAttribute(attributesToSplitOn, selectedAttribute);
			
			// make a branch for each value of the selected attribute
			double[] uniqueValues = features.getUniqueValuesArray(selectedAttribute);
			for (double value : uniqueValues)	{
				System.out.println("Branch value: " + value + " for attribute: " + selectedAttribute);
				// copy the matrices & create a sub-matrix
				Matrix features_copy = new Matrix(features);
				Matrix labels_copy = new Matrix(labels);
				Matrix.filterMatrixByAttributeValue(features_copy, labels_copy, selectedAttribute, value);
				// recurse and add the subtree as a child here
				DTNode child = induceTree(features_copy, labels_copy, deepCopyArrayList(attributesToSplitOn));	//give a deep copy of the arrayList to not change the original
				child.branchValue = value;
				root.addChild(child);		//attach the child branches to the root
			}
		}
		return root;
	}
		

	/*
	 * Makes a deep copy of an ArrayList of Integers
	 */
	private ArrayList<Integer> deepCopyArrayList(ArrayList<Integer> other)	{
		ArrayList<Integer> newArrayList = new ArrayList<Integer>();
		
		for(int i = 0; i < other.size(); i++)	{
			newArrayList.add(i, other.get(i));
		}
		
		return newArrayList;
	}
	
	/*
	 * Removes the attribute which we have already seen
	 */
	private void removeAttribute(ArrayList<Integer> attributesToSplitOn, int attribute)	{
		
		attributesToSplitOn.remove(new Integer(attribute));
		System.out.println("After Removing: " + attributesToSplitOn);
	}
	
	/*
	 * Prints out the arff matrix in the following format
	 * 	features => label
	 */
	public static void printMatrices(Matrix features, Matrix labels)	{
		// prints the numbers 
		for(int i = 0; i < features.rows(); i++)	{
			String row = "";
			for(int j = 0; j < features.row(i).length; j++)	{
				row += "  " + features.row(i)[j];
			}
			System.out.println(row + " => " + labels.row(i)[0]);
		}
		
		System.out.println("\n");
		
		// prints the values
		for(int i = 0; i < features.rows(); i++)	{
			String row = "";
			for(int j = 0; j < features.row(i).length; j++)	{
				int value = (int) features.row(i)[j];
				row += features.attrValue(j, value) + "\t";
			}
			int labelvalue = (int) labels.row(i)[0]; 
			System.out.println(row + " => " + labels.attrValue(0, labelvalue));//labels.row(i)[0]);
		}
//		System.out.println("\n");
	}
	
	/* 
	 * A feature vector goes in. A label vector comes out. (Some supervised
	 * learning algorithms only support one-dimensional label vectors. Some
	 * support multi-dimensional label vectors.)
	 */
	public void predict(double[] features, double[] labels) throws Exception {
		//TODO auto-generated stub
		
	}
	

	/*
	 *  Checks if all instances of the features Matrix
	 *  have the same classification in labels
	 *  
	 *  	Returns the label if they do
	 *  	Returns Double.NEGATIVE_INFINITY if they don't
	 */
	public static double checkIfAllElementsShareClass(Matrix features, Matrix labels)	{
		HashSet<Double> uniqueLabels = new HashSet<Double>();
		
		for(int i = 0; i < features.rows(); i++)	{
			uniqueLabels.add(labels.get(i,0));
		}
		
		// return the classification if they're the same
		if(uniqueLabels.size() == 1)	{
			double classification = labels.get(0, 0);
			return classification;
		} 
		else	{
			//if they're not all the same...
			return Double.NEGATIVE_INFINITY; 
		}
	}
	
	
	//---------------------------------------------------
	/*
	 * Node for the decision tree
	 */
	private class DTNode	{
		
		Matrix features;
		Matrix labels;
		int attribute; 	//each node represents an attribute which we branch off of there
		double branchValue;			//the value of the attribute (e.g., for attribute "Income Level" we have branches for "Low" "Med" and "High")
		double classification;			//if this is a leaf node, it has a classification and no children
		ArrayList<DTNode> children;
		
		/*
		 * Constructor
		 */
		public DTNode(Matrix features, Matrix labels)	{
			this.features = features;
			this.labels = labels;
			this.children = new ArrayList<DTNode>();
		}
		
		/*
		 * Returns true if this is a leaf node
		 */
		public boolean isLeafNode()	{
			if(children.size() == 0)	{
				return true;
			}
			else {
				return false;
			}
		}
		
		/*
		 * Adds a child DTNode
		 */
		public void addChild(DTNode child)	{
			this.children.add(child);
		}
		
		/*
		 * Prints the node and all of its children
		 *
		 *  tear-prod-rate = reduced: none
		 *		tear-prod-rate = normal
		 *		|  astigmatism = no
		 *	 	|  |  age = young: soft
		 *		|  |  age = pre-presbyopic: soft
		 *		|  |  age = presbyopic
		 *		|  |  |  spectacle-prescrip = myope: none
		 *		|  |  |  spectacle-prescrip = hypermetrope: soft
		 *		|  astigmatism = yes
		 *		|  |  spectacle-prescrip = myope: hard
		 *		|  |  spectacle-prescrip = hypermetrope
		 *		|  |  |  age = young: hard
		 *		|  |  |  age = pre-presbyopic: none
		 *		|  |  |  age = presbyopic: none
		 */
		public String toString()	{
			String toReturn = "";
			
			//print self
			toReturn += features.attrName(attribute) + " = " + features.attrValue(attribute, (int)branchValue);
			
			
			if(isLeafNode())	{
				toReturn += " : " + labels.attrValue(attribute,  (int)classification) + "\n"; 
			}
			else	{
				toReturn += "\n";
			}
			
			//test so far
			System.out.println(toReturn);
			
			//print children
			for(int i = 0; i < children.size(); i++)	{
				toReturn += "  | " + children.get(i).toString();
			}
			
			return toReturn;
		}
	}	//end DTNode class
	
	/*
	 * Never actually use this method - had to implement it for NeuralNet
	 */
	@Override
	public void train(Matrix trainFeatures, Matrix trainLabels,
			Matrix testFeatures, Matrix testLabels) {
		// TODO Auto-generated method stub
		
	}
	
	
	//---------------------------------------------------
	/*
	 *	Entropy
	 */
	private static class Entropy	{
		
		/*
		 * Returns the attribute from attributeList which gives the highest information gain
		 */
		public static int getHighestInformationGain(Matrix features, Matrix labels, ArrayList<Integer> attributeList)	{
			
			ArrayList<Double> informationGains = new ArrayList<Double>();
			for(int attrColumn = 0; attrColumn < attributeList.size(); attrColumn++)	{
				int attr = attributeList.get(attrColumn);
				double info_gain = computeInformationGain(features, labels, attr);
				informationGains.add(attrColumn, info_gain);
			}
			
			//return the index of the highest info_gain
			double max = informationGains.get(0);
			int index = 0;
			for(int i = 1; i < informationGains.size(); i++)	{
				if(informationGains.get(i) > max)	{
					max = informationGains.get(i);
					index = i;
				}
			}
			
//			System.out.println("Highest information gain is column: " + index + " which corresponds with " + attributeList.get(index) + " in AttributeList");
			return attributeList.get(index);
		}
		
		/*
		 * Calculates entropy
		 */
		public static double computeEntropy(Matrix features, Matrix labels)	{
			double entropy = 0.0;
			
			double[] uniqueLabels = labels.getUniqueValuesArray(0);
			for (double label : uniqueLabels)	{
				double instance_count = 0;
				for(int instanceNumber = 0; instanceNumber < labels.rows(); instanceNumber++)	{
					if(labels.get(instanceNumber, 0) == label)	{
						instance_count += 1;
					}
				}
//				System.out.println(instance_count + " instances of class #" + label);
				double probability = instance_count/labels.rows();
				entropy -= probability * (Math.log(probability)/Math.log(2));
//				System.out.println("\t|Entropy for class " + label + " is " + entropy);
			}
			return entropy;
		}
		
		/*
		 * Calculates information gain
		 * 		attrColumn is the column for which we will calculate information gain 
		 */
		public static double computeInformationGain(Matrix features, Matrix labels, int attrColumn)	{
			
			double features_entropy = computeEntropy(features, labels);
			double sum = 0.0;
			
			double[] uniqueValues = features.getUniqueValuesArray(attrColumn);
			for(double value : uniqueValues)	{
				//copy the matrix
				Matrix subFeatures = new Matrix(features);
				Matrix subLabels = new Matrix(labels);
				//filter for that value
				Matrix.filterMatrixByAttributeValue(subFeatures, subLabels, attrColumn, value);
				//compute the entropy
				double proportion = (double)subFeatures.rows() / (double)features.rows();
				double entropy_subfeatures = computeEntropy(subFeatures, subLabels);
				sum += proportion * entropy_subfeatures;
			}
			//calculate the information gain
			double information_gain = features_entropy - sum;
			System.out.println("Information gain for attrColumn #" + attrColumn + ": " + information_gain);
			return information_gain;
		}
		
	}	//end Entropy
	
}	//end DecisionTree class

