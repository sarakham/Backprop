import java.util.ArrayList;

/*
 * Node in the Neural Network Backpropagation Algorithm
 */
public class BPNode {
	public ArrayList<Double> weights;
	public ArrayList<Double> weightChanges;
	public double value;
	public double error;
	String type;
	
	/*
	 * Constructor
	 */
	public BPNode(String type)	{
		weightChanges = new ArrayList<Double>();
		weights = new ArrayList<Double>();
		value = 0;
		this.type = type;
	}
	
//	/*
//	 * Copy constructor
//	 */
//	BPNode(BPNode other)	{
//		//copy weights 
//		for(int i=0; i < other.weights.size(); i++)	{
//			this.weights.add(i, other.weights.get(i));
//		}
//		
//		//copy weightChanges
//		for(int i=0; i < other.weightChanges.size(); i++)	{
//			this.weightChanges.add(i, other.weightChanges.get(i));
//		}
//		
//		this.value = other.value;
//		this.error = other.error;
//		this.type = other.type;
//	}
	
	/*
	 * Returns a string object of the BPNode
	 */
	public String toString()	{
		String toReturn = "\n\t" + type + " node - value: " + value + "\n\t\tweights: ";
		for (int i = 0; i < weights.size(); i++)	{
			toReturn += " " + weights.get(i);
		}
		return toReturn;
	}
}
