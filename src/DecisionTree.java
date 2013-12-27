import java.util.ArrayList;
import java.util.HashSet;

public class DecisionTree extends SupervisedLearner {

	int NUM_ATTRIBUTES = 0;
	String DATA_TYPE = "discrete";
//	String DATA_TYPE = "continuous";
//	static String SPLITTING_CRITERIA = "information_gain";
	static String SPLITTING_CRITERIA = "accuracy";
	Matrix features_original;
	Matrix labels_original;
	DTNode rootNode;

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
		
		//remove unkonwn values
		printMatrices(features, labels);
//		features.removeUnknowns();
		
		// build tree
		rootNode = induceTree(features, labels, attributesToSplitOn);
//		String tree = rootNode.toString(null);
//		System.out.println("-------------");
//		System.out.println(tree);
	}

	
	/*
	 *  Create the tree
	 */
	public DTNode induceTree(Matrix features, Matrix labels, ArrayList<Integer> attributesToSplitOn)	{
//		System.out.println("--------------------\n Current Root Node"); printMatrices(features, labels);
//		System.out.println("attributeList: " + attributesToSplitOn);
		
		// Make node of the current tree
		DTNode root = new DTNode(features, labels);
		
		double unanimousClassification = checkIfAllElementsShareClass(features, labels);
		
		if(unanimousClassification >= 0)	{		// all instances have the same classification - return leaf node labeled with that class
			root.classification = unanimousClassification;
			root.setClassificationString();
			root.type = "Leaf";
//			System.out.println("\nLeaf node classification: " + root.classification);
//			System.out.println("-------------------------------\n");
		}
		else if(attributesToSplitOn.size() == 0)	{	//return a leaf node labeled as the majority class
			root.classification = labels.mostCommonValue(0);
			root.setClassificationString();
			root.type = "leaf";
//			System.out.println("\nLeaf node classification is majority class: " + root.classification);
//			System.out.println("-------------------------------\n");
		}
		else	{
			// find where to split
			String[] accuracyStillImproving = new String[1];
			int selectedAttribute = Entropy.getHighestInformationGain(features, labels, attributesToSplitOn, accuracyStillImproving);
			root.attribute = selectedAttribute;
			root.setAttributeName();
//			System.out.println("selectedAttributeColumn: " + selectedAttribute + " (" + features.attrName(selectedAttribute) + ")  unanimousClassification: " + unanimousClassification);
			removeAttribute(attributesToSplitOn, selectedAttribute);
			
			if(accuracyStillImproving[0] == "no")	{
				//we are no longer improving accuracy and need to create a leaf node with the majority
				root.classification = labels.mostCommonValue(0);
				root.setClassificationString();
				root.type = "Leaf";
//				System.out.println("\nLeaf node classification: " + root.classification);
//				System.out.println("-------------------------------\n");
			}
			else if(isContinuous(features) == false)	{
				// make a branch for each value of the selected attribute
				double[] uniqueValues = features.getUniqueValuesArray(selectedAttribute);
				for (double value : uniqueValues)	{
//					System.out.println("Branch value: " + value + " for attribute: " + selectedAttribute);
					// copy the matrices & create a sub-matrix
					Matrix features_copy = new Matrix(features);
					Matrix labels_copy = new Matrix(labels);
					Matrix.filterMatrixByAttributeValue(features_copy, labels_copy, selectedAttribute, value);
					// recurse and add the subtree as a child here
					DTNode child = induceTree(features_copy, labels_copy, deepCopyArrayList(attributesToSplitOn));	//give a deep copy of the arrayList to not change the original
					if(child.isLeafNode())	{
						child.attribute = selectedAttribute;
						child.setAttributeName();
					}
					child.branchValue = value;
					child.setBranchValueString(selectedAttribute);
					root.addChild(child);
				}
			}
			else	{
				// branch for values below the mean and above the mean
				double mean = features.columnMean(selectedAttribute);
//				System.out.println("Mean: " + mean);
				
				// copy the matrices & create a sub-matrix BELOW MEAN
				Matrix features_below_mean = new Matrix(features);
				Matrix labels_below_mean = new Matrix(labels);
				Matrix.filterMatrixByLessThanMeanValue(features_below_mean, labels_below_mean, selectedAttribute);
				//recurse and add the subtree as a child here
				DTNode left_child = induceTree(features_below_mean, labels_below_mean, deepCopyArrayList(attributesToSplitOn));
				if(left_child.isLeafNode())	{
					left_child.attribute = selectedAttribute;
				}
				left_child.branchValue = 0;
				root.addChild(left_child);
					
				// copy the matrices & create a sub-matrix ABOVE MEAN
				Matrix features_above_mean = new Matrix(features);
				Matrix labels_above_mean = new Matrix(labels);
				Matrix.filterMatrixByGreaterThanMeanValue(features_above_mean, labels_above_mean, selectedAttribute);
				//recurse and add the subtree as a child here
				DTNode right_child = induceTree(features_above_mean, labels_above_mean, deepCopyArrayList(attributesToSplitOn));
				if(right_child.isLeafNode())	{
					right_child.attribute = selectedAttribute;
				}
				right_child.branchValue = 1;
				root.addChild(right_child);
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
//		System.out.println("After Removing: " + attributesToSplitOn);
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
//			System.out.println(row + " => " + labels.row(i)[0]);
		}
		
//		System.out.println("\n");
		
		// prints the values
		if(!isContinuous(features))	{
			for(int i = 0; i < features.rows(); i++)	{
				String row = "";
				for(int j = 0; j < features.row(i).length; j++)	{
					int value = (int) features.row(i)[j];
					row += features.attrValue(j, value) + "\t";
				}
				int labelvalue = (int) labels.row(i)[0]; 
//				System.out.println(row + " => " + labels.attrValue(0, labelvalue));//labels.row(i)[0]);
			}
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
		double classification = traverse(features, rootNode);
		labels[0] = classification;
	}
	
	/*
	 * Traverses the tree to find the prediction value
	 */
	public double traverse(double[] features, DTNode cur)	{
		
		if(cur.isLeafNode())	{
			return cur.classification; 
		}
		else	{
			//determine which way to go
			if(DATA_TYPE == "continuous")	{
				// find the attribute value corresponding to the currentNode
				double attrValue = features[cur.attribute];
				//traverse down the right branch
				double mean = cur.features.columnMean((int)attrValue);
				if(features[(int)attrValue] < mean)	{
					return traverse(features, cur.children.get(0));
				}
				else	{
					return traverse(features, cur.children.get(1));
				}
			}
			else	{	//discrete
				// find the attribute value corresponding to the currentNode
				double attrValue = features[cur.attribute];
				// traverse the right branch 
//				System.out.println("attrValue: " + attrValue);
				if(cur.children.size() == 2){
					return traverse(features, cur.children.get((int)attrValue));
				}
				else	{
					return traverse(features, cur.children.get(0));
				}
			}
		}
	}
	
	/*
	 * Returns true if all elements in an ArrayList<Double> are the same
	 */
	public static boolean checkIfElementsAreSame(ArrayList<Double> list)	{
		
		if(list.size() > 1)	{
			double firstVal = list.get(0);
			for(double d : list)	{
				if (d != firstVal)	{	
					return false;		
				}
			}
			return true;
		}
		else	{
			//list is empty or only has one element
			return true;
		}
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
	
	/*
	 * Check if continuous attribute
	 */
	public static boolean isContinuous(Matrix features)	{
		if (features.m_enum_to_str.get(0).size() == 0)	{
			return true;
		}
		return false;
	}

	/*
	 * Checks if all elements in the ArrayList are negative
	 */
	public static boolean checkIfAllNegative(ArrayList<Double> list)	{
		
		for(double d : list)	{
			if (d >= 0)	{	
				return false;		
			}
		}
		return true;
	}
	
	//---------------------------------------------------
	/*
	 * Node for the decision tree
	 */
	private class DTNode	{
		
		Matrix features;
		Matrix labels;
		int attribute; 				//each node represents an attribute which we branch off of there
		double branchValue;			//the value of the attribute (e.g., for attribute "Income Level" we have branches for "Low" "Med" and "High")
		double classification;			//if this is a leaf node, it has a classification and no children
		ArrayList<DTNode> children;
		String type;
		String branchValueString;
		String attributeName;
		String classificationValue;
		
		/*
		 * Constructor
		 */
		public DTNode(Matrix features, Matrix labels)	{
			this.features = features;
			this.labels = labels;
			this.children = new ArrayList<DTNode>();
			this.attribute = -1;
			this.branchValue = -1;
			this.classification = -1;
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
		 * Finds the String value of the classification for printing out
		 */
		public void setClassificationString()	{
			this.classificationValue = labels.attrValue(0, (int)classification);
		}
		
		/*
		 * Sets the string for the branch value, which is an attribute value from the parent's attribute, not the child
		 */
		public void setBranchValueString(int parentAttribute)	{
			if(isContinuous(features) == false)	{
				this.branchValueString = features.attrValue(parentAttribute, (int)branchValue);
			}
		}
		
		/*
		 * Sets the string for the attribute
		 */
		public void setAttributeName()	{
			this.attributeName = features.attrName(attribute);
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
		public String toString(String prefix)	{
			String out = "";

			//get all children branch values
			ArrayList<Integer> branchValues = new ArrayList<Integer>();
			for(int i = 0; i < children.size(); i++)	{
				branchValues.add((int)children.get(i).branchValue);
			}
			
			if(isContinuous(features) == false)	{
				if(branchValue < 0)	{	// is the root node
					// just print children
					for(int i = 0; i < children.size(); i++)	{
	
						if(children.get(i).isLeafNode())	{
							out += children.get(i).attributeName + " = " + children.get(i).branchValueString + " : " + children.get(i).classificationValue + "\n";
	//						System.out.println(children.get(i).attributeName + " = " + children.get(i).branchValueString + " : " + children.get(i).classificationValue + "\n");
						}
						else	{
							out += attributeName + " = " + children.get(i).branchValueString + "\n";
	//						System.out.println(attributeName + " = " + children.get(i).branchValueString + "\n");
							out += children.get(i).toString("| ");
						}
					}	//end for
				}
				else	{	//not the root node
					if(isLeafNode() == false)	{
						//print children
						for(int i = 0; i < children.size(); i++)	{
							out += prefix + attributeName + " = " + children.get(i).branchValueString;
	//						System.out.println(prefix + attributeName + " = " + children.get(i).branchValueString + "\n");
							
							if(children.get(i).isLeafNode())	{	//add classification
								out += " : " + children.get(i).classificationValue + "\n";
							}
							else	{
								out += "\n" + children.get(i).toString(prefix + "| ");
							}
						}
					}
				}
			}
			else	{	//continuous
				if(branchValue < 0)	{	// is the root node
					// just print children
					for(int i = 0; i < children.size(); i++)	{
	
						if(children.get(i).isLeafNode())	{
//							out += children.get(i).attributeName + " = " + children.get(i).branchValueString + " : " + children.get(i).classificationValue + "\n";
							out += "Node: property = " + children.get(i).attributeName + " : " + children.get(i).classificationValue + "\n";
//							System.out.println("Node: property = " + children.get(i).attributeName + " : " + children.get(i).classificationValue + "\n");
						}
						else	{
//							out += attributeName + " = " + children.get(i).branchValueString + "\n";
							out += "Node: property = " + attributeName + "\n";
//							System.out.println("Node: property = " + attributeName + "\n");
							out += children.get(i).toString("| ");
						}
					}	//end for
				}
				else	{	//not the root node
					if(isLeafNode() == false)	{
						//print children
						for(int i = 0; i < children.size(); i++)	{
//							out += prefix + attributeName + " = " + children.get(i).branchValueString;
							out += prefix + "Node: property = " + attributeName;
//							System.out.println(prefix + "Node: property = " + attributeName);
							
							if(children.get(i).isLeafNode())	{	//add classification
//								out += " : " + children.get(i).classificationValue + "\n";
								out += " : " + children.get(i).classificationValue + "\n";
							}
							else	{
								out += "\n" + children.get(i).toString(prefix + "| ");
							}
						}
					}
				}
			}
			
			return out;
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
		 * Returns the attribute from attributeList which gives the highest information gain or accuracy depending on what the splitting criteria are
		 */
		public static int getHighestInformationGain(Matrix features, Matrix labels, ArrayList<Integer> attributeList, String[] accuracyStillImproving)	{
			
			ArrayList<Double> informationGains = new ArrayList<Double>();
			for(int attrColumn = 0; attrColumn < attributeList.size(); attrColumn++)	{
				int attr = attributeList.get(attrColumn); 	//only use columns in the attributeList (prevents repeats)
				double info_gain = 0.0;
				
				if(SPLITTING_CRITERIA == "accuracy")	{
					
//					System.out.println("Splitting on accuracy");
					
					if(isContinuous(features))	{
						double accuracy = computeAccuracy(features, labels);
						// data set below mean
						Matrix features_below_mean = new Matrix(features);
						Matrix labels_below_mean = new Matrix(labels);
						// compute entropy & proportion
						Matrix.filterMatrixByLessThanMeanValue(features_below_mean, labels_below_mean, attrColumn);
						double proportion_below = (double)features_below_mean.rows() / (double)features.rows();
						double accuracy_below = computeAccuracy(features_below_mean, labels_below_mean);
						
						// data set above mean
						Matrix features_above_mean = new Matrix(features);
						Matrix labels_above_mean = new Matrix(labels);
						// compute entropy & proportion
						Matrix.filterMatrixByGreaterThanMeanValue(features_above_mean, labels_above_mean, attrColumn);
						double proportion_above = (double)features_above_mean.rows() / (double)features.rows();
						double accuracy_above = computeAccuracy(features_above_mean, labels_above_mean);
						
						double information_gain = ((proportion_above * accuracy_above) * (proportion_below * accuracy_below)) - accuracy;
						informationGains.add(attrColumn, information_gain);
					}
					else	{
						double accuracy = computeAccuracy(features, labels);
						//loop through the values and compute the accuracies of each 
						double[] uniqueValues = features.getUniqueValuesArray(attrColumn);
						double sum = 0.0;
						for (double value : uniqueValues)	{
							//copy the matrix
							Matrix features_copy = new Matrix(features);
							Matrix labels_copy = new Matrix(labels);
							//filter by the value
							Matrix.filterMatrixByAttributeValue(features_copy, labels_copy, attrColumn, value);
							printMatrices(features_copy, labels_copy);
							//compute the accuracy
							double proportion = (double)features_copy.rows() / (double)features.rows();
							double features_copy_accuracy = computeAccuracy(features_copy, labels_copy);
							sum += proportion * features_copy_accuracy;
						}
						info_gain = sum - accuracy; 
						informationGains.add(attrColumn, info_gain);
					}
				}
				else	{	//information gain
					if(isContinuous(features))	{
						info_gain = computeInformationGainContinuous(features, labels, attr);
					}
					else {
						info_gain = computeInformationGain(features, labels, attr);
					}
					informationGains.add(attrColumn, info_gain);
				}
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
			
			//check if all the gains were equal (this is for accuracy to check whether it's still improving
			if(SPLITTING_CRITERIA == "accuracy")	{
				if(checkIfElementsAreSame(informationGains) || checkIfAllNegative(informationGains))	{
					accuracyStillImproving[0] = "no";
				}
			}
			
//			System.out.println("Highest information gain is column: " + index + " which corresponds with " + attributeList.get(index) + " in AttributeList");
//			System.out.println("InfoGains: " + informationGains);
			return attributeList.get(index);
		}
		
		/*
		 * Computes the accuracy
		 * 		Returns the number of instances that have the majority classification
		 */
		public static double computeAccuracy(Matrix features, Matrix labels)	{
			
			double majorityClassification = labels.mostCommonValue(0);
			int instanceCount = 0;
			
			for(int i = 0; i < labels.rows(); i++)	{
				if(labels.get(i, 0) == majorityClassification)	{
					instanceCount++;
				}
			}
			
			return (double)instanceCount / (double)labels.rows();
		}
		
		
		/*
		 * Calculates entropy
		 */
		public static double computeEntropy(Matrix features, Matrix labels)	{
			double entropy = 0.0;
			
			double[] classes = labels.getUniqueValuesArray(0);
			for (double label : classes)	{
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
		 * Calculates information gain of discrete data
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
//			System.out.println("Information gain for attrColumn #" + attrColumn + ": " + information_gain);
			return information_gain;
		}
		
		
		/*
		 * Computes the Information Gain of continuous data
		 */
		public static double computeInformationGainContinuous(Matrix features, Matrix labels, int attrColumn)	{
			double features_entropy = computeEntropy(features, labels);
			
			// data set below mean
			Matrix features_below_mean = new Matrix(features);
			Matrix labels_below_mean = new Matrix(labels);
			// compute entropy & proportion
			Matrix.filterMatrixByLessThanMeanValue(features_below_mean, labels_below_mean, attrColumn);
			double proportion_below = (double)features_below_mean.rows() / (double)features.rows();
			double entropy_below = computeEntropy(features_below_mean, labels_below_mean);
			
			// data set above mean
			Matrix features_above_mean = new Matrix(features);
			Matrix labels_above_mean = new Matrix(labels);
			// compute entropy & proportion
			Matrix.filterMatrixByGreaterThanMeanValue(features_above_mean, labels_above_mean, attrColumn);
			double proportion_above = (double)features_above_mean.rows() / (double)features.rows();
			double entropy_above = computeEntropy(features_above_mean, labels_above_mean);
			
			double information_gain = features_entropy - ((proportion_above * entropy_above) * (proportion_below * entropy_below)); 
			return information_gain;
		}
		
	}	//end Entropy
	
}	//end DecisionTree class

