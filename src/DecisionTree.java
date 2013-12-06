import java.text.AttributedCharacterIterator.Attribute;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.TreeMap;

public class DecisionTree extends SupervisedLearner {

	int NUM_ATTRIBUTES = 0;
	String SPLITTING_CRITERION;
	Matrix features_original;
	Matrix labels_original;

	/*
	 * Constructor
	 */
	public DecisionTree()	{
		predictErrors = new ArrayList<Double>();
	}
	
	/* 
	 * 	Train
	 */
	public void train(Matrix features, Matrix labels) throws Exception {
		
		NUM_ATTRIBUTES = features.cols();
		features_original = new Matrix(features);
		labels_original = new Matrix(labels);
		
		printMatrices(features, labels);

		// list of attributes to split on
		ArrayList<Integer> attributesToSplitOn = new ArrayList<Integer>();
		for(int attribute = 0; attribute < NUM_ATTRIBUTES; attribute++)	{
			attributesToSplitOn.add(attribute);
		}
		
		DTNode root = new DTNode(features, labels);
		root.children.add(new DTNode(features, labels));
		root.children.add(new DTNode(features, labels));
		root.children.add(new DTNode(features, labels));
		root.children.get(0).children.add(new DTNode(features, labels));
		root.children.get(0).children.add(new DTNode(features, labels));
		root.children.get(2).children.add(new DTNode(features, labels));
		
		System.out.println("Print");
		System.out.println(root);
		
		//begin building the tree
//		Node root = induceTree(features, labels, attributesToSplitOn);
//		System.out.println("Print root");
//	//	printTree(root, "");
	}

	
	/*
	 *  Create the tree
	 */
	public Node induceTree(Matrix features, Matrix labels, ArrayList<Integer> attributesToSplitOn)	{
		
		int selectedAttributeColumn = Entropy.getHighestInfoGain(features, labels, attributesToSplitOn);
		double unanimousClassification = checkIfAllElementsShareClass(features, labels);
		
		System.out.println("selectedAttributeColumn: " + selectedAttributeColumn + "  unanimousClassification: " + unanimousClassification);

		if(unanimousClassification >= 0)	{
			//return a leaf node labeled with that class
		}
		else if(attributesToSplitOn.isEmpty())	{
			//return a leaf node labeled with the majority class in Example-set
			Node leaf = new Node();
			
			leaf.setClassificationIndex((int)unanimousClassification);
//			leaf.setAttributeIndex(selectedAttributeColumn);
//			leaf.setFeatures(features);
//			leaf.setLabels(labels);
//			System.out.println("\n************************");	//REMOVE
//			System.out.println("NEW LEAF NODE with class index: " + result + " (" + labels.attrValue(0, result) + ")");
//			leaf.print("");
//			System.out.println("************************");
//			
//		return leaf;
		}
		else	{
//			Select P from Properties  (*)
//			Remove P from Properties
//			Make P the root of the current tree
//			For each value V of P
//				Create a branch of the current tree labeled by V
//				Partition_V <- Elements of Example-set with value V for P
//				Induce-Tree(Partition_V, Properties)
//				Attach result to branch V
		}
		
//		if(result >= 0)	{				//if all elements of features[bestAttributeColumn] have the same class
//			Node leaf = new Node();		//return a leaf node associated with the class
//				leaf.setClassificationIndex(result);
//				leaf.setAttributeIndex(selectedAttributeColumn);
//				leaf.setFeatures(features);
//				leaf.setLabels(labels);
//				System.out.println("\n************************");	//REMOVE
//				System.out.println("NEW LEAF NODE with class index: " + result + " (" + labels.attrValue(0, result) + ")");
//				leaf.print("");
//				System.out.println("************************");
//				
//			return leaf;
//		}
//		else if(unusedAttributes.size() == 0)	{	//return a leaf node labeled as the majority class
//			Node leaf = new Node();
//				double label = labels.mostCommonValue(0);	//should only have rows for this attribute left
//				leaf.setClassificationIndex((int)label);				//TODO make this save the index, not the value
//				leaf.setAttributeIndex(selectedAttributeColumn);
//				leaf.setFeatures(features);
//				leaf.setLabels(labels);
//				System.out.println("\n************************");	//REMOVE
//				System.out.println("NEW LEAF NODE with class index: " + result + " (" + labels.attrValue(0, result) + ")");
//				leaf.print("");
//				System.out.println("************************");
//			return leaf;
//		}
//		else	{
//			unusedAttributes.remove(new Integer(selectedAttributeColumn));
//			Node root = new Node();
//				root.setAttributeIndex(selectedAttributeColumn);
//				root.setFeatures(features);
//				root.setLabels(labels);
//			
//			//make a branch for each attribute value
//			double[] uniqueValues = features.getUniqueValuesArray(selectedAttributeColumn);
//			for(int attrValue = 0; attrValue < uniqueValues.length; attrValue++) {
//					Matrix features_partition = new Matrix(features);
//					Matrix labels_partition = new Matrix(labels);
//					Matrix.selectAttributeValue(features_partition, labels_partition, selectedAttributeColumn, uniqueValues[attrValue]);
//					features_partition.print();	//REMOVE
//					labels_partition.print();	//REMOVE
//					
//					//set the value of that branch's node to be the selected feature
//					System.out.println("Branch's value: " + features_partition.attrValue(selectedAttributeColumn, attrValue));	//REMOVE
//					Node child = induceTree(features_partition, labels_partition, unusedAttributes);
//					child.setAttributeIndex(selectedAttributeColumn);
//					child.setBranchValue(attrValue);
//					System.out.println("TESTING BRANCH VALUE: " + child.getBranchValueName());
//					System.out.println("TESTING EQUIVALENCE: " + (selectedAttributeColumn == child.getAttributeIndex()));
//					
//					root.addChild(child);
//			}
////			printTree(root,"");
//			return root;
//		}
		return null;
	}
	
	
	/*
	 * Prints out the arff matrix in the following format
	 * 	features => label
	 */
	private void printMatrices(Matrix features, Matrix labels)	{
		// print the matrix
		System.out.println("Training");
		for(int i = 0; i < features.rows(); i++)	{
			String row = "";
			for(int j = 0; j < features.row(i).length; j++)	{
				row += "  " + features.row(i)[j];
			}
			System.out.println(row + " => " + labels.row(i)[0]);
		}
		
		System.out.println("median: " + labels.columnMedian(0));
		System.out.println("mean: " + labels.columnMean(0));
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
	 * Recursively prints all the nodes
	 */
	public void printTree(Node node, String indent)	{
		
		//TODO print the tree
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
	 * Node for the decision tree
	 */
	private class DTNode	{
		
		Matrix features;
		Matrix labels;
		int attributeIndex; 	//the attribute the parent of this node branched on
		int attributeValue;		//the value of the attribute
		int classification;		//if this is a leaf node
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
			//TODO print the node and all of its children
			String toReturn = "";
			
			//print self
			toReturn += "Attribute name" + " = " + "attributeValue";
			if(isLeafNode())	{
				toReturn += " : " + "classification\n"; 
			}
			else	{
				toReturn += "\n";
			}
			
			//print children
			for(int i = 0; i < children.size(); i++)	{
				toReturn += "\t| " + children.get(i).toString();
			}
			
			return toReturn;
		}
	}	//end DTNode class
	
}	//end DecisionTree class
