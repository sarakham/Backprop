import java.util.ArrayList;

/*
 * Node in the Neural Network Backpropagation Algorithm
 */
public class BPNode {
	public ArrayList<Double> weights;
	public ArrayList<Double> weightChanges;
	public double value;
	String type;
	
	public BPNode(String type)	{
		weightChanges = new ArrayList<Double>();
		weights = new ArrayList<Double>();
		value = 0;
		this.type = type;
	}
	
	public String toString()	{
		String toReturn = "\n\t" + type + " node - value: " + value + "\n\t\tweights: ";
		for (int i = 0; i < weights.size(); i++)	{
			toReturn += " " + weights.get(i);
		}
		return toReturn;
	}
}
