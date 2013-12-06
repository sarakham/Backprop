import java.util.ArrayList;

/*
 * Node object to use in building a decision tree
 * 		In the case of calculating Information Gain, the children's values will be classifications
 * 		In the case of calculating Accuracy, __________________________
 */
public class Node	{
	
	ArrayList<Node> children;
	Matrix features;
	Matrix labels;
	int attributeIndex;		//the column of Node's attribute in features
	int labelIndex;			//the value a leaf node will return
	int branchValue;
	
	/*
	 *  Constructor
	 */
	public Node()	{
		this.children = new ArrayList<Node>();
		this.features = null;
		this.labels = null;
		this.attributeIndex = -1;		//the attribute this node represents
		this.labelIndex = -1;
		this.branchValue = -1;			//the value branched on, leading to this node
	}
	
	/*
	 *  Returns the column number corresponding with this node's attribute
	 */
	public int getAttributeIndex() {
		return attributeIndex;
	}

	/*
	 *  Sets the column number corresponding with this node's attribute
	 */
	public void setAttributeIndex(int attributeIndex) {
		this.attributeIndex = attributeIndex;
	}
	
	/*
	 *  Returns the name of the attribute, if it has been set.  If not, it returns null
	 */
	public String getAttributeName()	{
		if(this.attributeIndex >= 0)	{
			return features.attrName(attributeIndex);
		}
		else	{
			return null;
		}
	}
	
	/*
	 *  Returns the index (the row) of the Labels matrix where the value is found
	 */
	public double getClassificationIndex()	{
		return labelIndex;
	}
	
	/*
	 *  Sets the index (the row) of the Labels matrix where the value is found
	 */
	public void setClassificationIndex(int label)	{
		this.labelIndex = label;
	}
	
	/*
	 *  Returns a string of the node's classification
	 *  if it's a leaf node, otherwise returns null 
	 */
	public String getClassificationName()	{
		if(isLeafNode())	{
			return labels.attrValue(0, labelIndex);
		}
		else	{
			return null;
		}
	}
	
	/*
	 * Returns an ArrayList of the child Nodes
	 */
	public ArrayList<Node> getChildren()	{
		return children;
	}
	
	/*
	 * Adds another child node
	 */
	public void addChild(Node child)	{
		this.children.add(child);
	}

	
	/*
	 *  Returns true if the Node is a leaf node 
	 *  (has no children and has a value)
	 */
	public boolean isLeafNode()	{
		if(children.size() > 0)	{
			return false;
		}
		else	{
			return true;
		}
	}

	public Matrix getFeatures() {
		return features;
	}

	public void setFeatures(Matrix features) {
		this.features = features;
	}

	public Matrix getLabels() {
		return labels;
	}

	public void setLabels(Matrix labels) {
		this.labels = labels;
	}
	

	/*
	 *  Returns the value of the branch which led to this node from the parent node
	 */
	public int getBranchValue() {
		return branchValue;
	}

	/*
	 *  Sets the value of the branch which led to this node form the parent node
	 */
	public void setBranchValue(int branchValue) {
		this.branchValue = branchValue;
	}
	
	/*
	 *  Returns the String value of the branch which led to this node from the parent node
	 */
	public String getBranchValueName()	{
		if(getBranchValue() > 0)	{
			String name = features.attrValue(getAttributeIndex(), getBranchValue()); 
			return name;
		}
		else	{
			return null;
		}
	}
	
	/* 
	 * Prints the node
	 */
	public void print(String indent)	{
		if(getBranchValue() >=0 )	{		// not the root node
			indent += "\t";
			System.out.println(indent + "Branch leading here: " + getBranchValueName());
			indent += "\t";
		}
		if(isLeafNode())	{
			System.out.println(indent + "Node classification: " + this.getClassificationName());
		}
		else	{
			System.out.println(indent + "Node Attribute: " + this.getAttributeName());
		}
	}
	
	//--------------------------------------------
//	/*
//	 *  Returns true if the Node has any child Nodes
//	 */
//	public boolean hasChildren()	{
//		return children.size() > 0;
//	}	
}
