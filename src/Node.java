import java.util.ArrayList;

/*
 * In the case of calculating Information Gain, the children's values will be classifications
 */
public class Node	{
	private ArrayList<Node> children;
	private Matrix features;
	private Matrix labels;
	private int attributeIndex;		//the column of Node's attribute in features
	private int labelIndex;		//the value a leaf node will return
	private int branchValue;
	
	// creates a blank Node
	public Node()	{
		this.setAttributeIndex(-1);	//setting as 0 would be misleading
		this.labelIndex = -1;
		this.setFeatures(null);
		this.setLabels(null);
		this.setBranchValue(-1);
		this.children = new ArrayList<Node>();
	}
	
	// Returns the column number corresponding with this node's attribute
	public int getAttributeIndex() {
		return attributeIndex;
	}

	// Sets the column number corresponding with this node's attribute
	public void setAttributeIndex(int attributeIndex) {
		this.attributeIndex = attributeIndex;
	}
	
	// Returns the name of the attribute, if it has been set.  If not, it returns null
	public String getAttributeName()	{
		if(this.attributeIndex >= 0)	{
			return features.attrName(attributeIndex);
		}
		else	{
			return null;
		}
	}
	
	// Returns the index (the row) of the Labels matrix where the value is found
	public double getClassificationIndex()	{
		return labelIndex;
	}
	
	// Sets the index (the row) of the Labels matrix where the value is found
	public void setClassificationIndex(int label)	{
		this.labelIndex = label;
	}
	
	// Returns a string of the node's classification
	//		should only call this on leaf nodes
	public String getClassificationName()	{
		return labels.attrValue(0, labelIndex);
	}
	
	public ArrayList<Node> getChildren()	{
		return children;
	}
	
	public void addChild(Node child)	{
		this.children.add(child);
	}
	
	// Returns true if the Node has any child Nodes
	public boolean hasChildren()	{
//		System.out.println("Number of children: " + children.size());
		return children.size() > 0;
	}
	
	// Returns true if the Node is a leaf node (has no children and has a value)
	public boolean isLeafNode()	{
		if(hasChildren() == true)	{
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
	

	// Returns the value of the branch which led to this node from the parent node
	public int getBranchValue() {
		return branchValue;
	}

	// Sets the value of the branch which led to this node form the parent node
	public void setBranchValue(int branchValue) {
		this.branchValue = branchValue;
	}
	
	// Returns the String value of the branch which led to this node from the parent node
	public String getBranchValueName()	{
		if(getBranchValue() > 0)	{
			String name = features.attrValue(getAttributeIndex(), getBranchValue()); 
			return name;
		}
		else	{
			return null;
		}
	}
	
	//print the node
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
}
