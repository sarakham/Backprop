import java.text.AttributedCharacterIterator.Attribute;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.TreeMap;

public class DecisionTreeLearner extends SupervisedLearner {

	private int numAttributes = 0;
	Matrix features_original;
	Matrix labels_original;

	/* 
	 * Before you call this method, you need to divide your data
	 * into a feature matrix and a label matrix.
	 */
	public void train(Matrix features, Matrix labels) throws Exception {
		
		numAttributes = features.cols();
		features_original = new Matrix(features);
		labels_original = new Matrix(labels);
		
		//my test code here
		System.out.println("---FEATURES---");
		features.print();
		System.out.println("---LABELS---");
		labels.print();
		//end test code
		
		//print features
		features.printValues("---FEATURES' VALUES---");
		labels.printValues("---LABELS' VALUES---");
		
		System.out.println("median: " + labels.columnMedian(0));
		System.out.println("mean: " + labels.columnMean(0));
		
		//------------------------
		
		// Add all the attributes to the list 
		ArrayList<Integer> unusedAttributes = new ArrayList<Integer>();
//		HashSet<Integer> unusedAttributes = new HashSet<Integer>();
		for(int attribute=0; attribute < numAttributes; attribute++)	{
			System.out.println("Adding: " + features.attrName(attribute));
			unusedAttributes.add(attribute);
		}
		
		//begin building the tree
		Node root = induceTree(features, labels, unusedAttributes);
		System.out.println("Print root");
	//	printTree(root, "");
	}

	/* 
	 * A feature vector goes in. A label vector comes out. (Some supervised
	 * learning algorithms only support one-dimensional label vectors. Some
	 * support multi-dimensional label vectors.)
	 */
	public void predict(double[] features, double[] labels) throws Exception {
//		for(int i = 0; i < m_labels.length; i++)
//			labels[i] = m_labels[i];
	}

	
	/*
	 *  Create the tree
	 */
	public Node induceTree(Matrix features, Matrix labels, ArrayList<Integer> unusedAttributes)	{
		
		int selectedAttributeColumn = Entropy.getHighestInfoGain(features, labels, unusedAttributes);
		int result = (int)doElementsShareClass(features, labels, selectedAttributeColumn);	//this returns a double

		if(selectedAttributeColumn != -1)	{
			System.out.println("\n----------------------");
			System.out.println("Selected Attribute: " + selectedAttributeColumn + " (" + features.attrName(selectedAttributeColumn) + ")");	//REMOVE
		}
		
		if(result >= 0)	{				//if all elements of features[bestAttributeColumn] have the same class
			Node leaf = new Node();		//return a leaf node associated with the class
				leaf.setClassificationIndex(result);
				leaf.setAttributeIndex(selectedAttributeColumn);
				leaf.setFeatures(features);
				leaf.setLabels(labels);
				System.out.println("\n************************");	//REMOVE
				System.out.println("NEW LEAF NODE with class index: " + result + " (" + labels.attrValue(0, result) + ")");
				leaf.print("");
				System.out.println("************************");
				
			return leaf;
		}
		else if(unusedAttributes.size() == 0)	{	//return a leaf node labeled as the majority class
			Node leaf = new Node();
				double label = labels.mostCommonValue(0);	//should only have rows for this attribute left
				leaf.setClassificationIndex((int)label);				//TODO make this save the index, not the value
				leaf.setAttributeIndex(selectedAttributeColumn);
				leaf.setFeatures(features);
				leaf.setLabels(labels);
				System.out.println("\n************************");	//REMOVE
				System.out.println("NEW LEAF NODE with class index: " + result + " (" + labels.attrValue(0, result) + ")");
				leaf.print("");
				System.out.println("************************");
			return leaf;
		}
		else	{
			unusedAttributes.remove(new Integer(selectedAttributeColumn));
			Node root = new Node();
				root.setAttributeIndex(selectedAttributeColumn);
				root.setFeatures(features);
				root.setLabels(labels);
			
			//make a branch for each attribute value
			double[] uniqueValues = features.getUniqueValuesArray(selectedAttributeColumn);
			for(int attrValue = 0; attrValue < uniqueValues.length; attrValue++) {
					Matrix features_partition = new Matrix(features);
					Matrix labels_partition = new Matrix(labels);
					Matrix.selectAttributeValue(features_partition, labels_partition, selectedAttributeColumn, uniqueValues[attrValue]);
					features_partition.print();	//REMOVE
					labels_partition.print();	//REMOVE
					
					//set the value of that branch's node to be the selected feature
					System.out.println("Branch's value: " + features_partition.attrValue(selectedAttributeColumn, attrValue));	//REMOVE
					Node child = induceTree(features_partition, labels_partition, unusedAttributes);
					child.setAttributeIndex(selectedAttributeColumn);
					child.setBranchValue(attrValue);
					System.out.println("TESTING BRANCH VALUE: " + child.getBranchValueName());
					System.out.println("TESTING EQUIVALENCE: " + (selectedAttributeColumn == child.getAttributeIndex()));
					
					root.addChild(child);
			}
//			printTree(root,"");
			return root;
		}
	}
	
	/*
	 * Recursively prints all the nodes
	 */
	public void printTree(Node node, String indent)	{
		
		if(node.isLeafNode())	{						//leaf
			node.print(indent);
		}
		else {											//not a leaf node
			node.print(indent);
			indent += "\t";
			for(Node child : node.getChildren())	{
				printTree(child, indent);
			}
		}		
	}
	
	//-----------------------------------------------------

	/*
	 *  Checks if all values of an attribute (feature) result in the same label
	 *  Returns the label if they do
	 *  Returns Double.NEGATIVE_INFINITY if they don't
	 */
	public static double doElementsShareClass(Matrix features, Matrix labels, int attributeColumn)	{
		// There should only be one attribute value represented in the column
		HashSet<Double> labelsAssigned = new HashSet<Double>();
		
		for(int i = 0; i < features.rows(); i++)	{
			labelsAssigned.add(labels.get(i,0));
		}
		
		if(labelsAssigned.size() == 1)	{
			//if they're all the same, return the top value in the attributeColumn
			return labels.get(0, 0);
		} 
		else	{
			//if they're not all the same...
			return Double.NEGATIVE_INFINITY; 
		}
	}
}
