import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Entropy {

	/*
	 * Calculates the entropy for the given attribute and the associated labels - was called calClEntropyForNom
	 */
	public static double calculateEntropyNominal (double[] colValues, double[] colUniqVals ){
		double entropy = 0;
		double probability = 0;
		int count =0;	

		int numInst= colValues.length;
		if(colValues.length == 0) {
			return 0; //no instances to consider
		}

		for(int colValUcnt=0;  colValUcnt<colUniqVals.length ; colValUcnt++) {
			count =0;			
			double colUniqValue = colUniqVals[colValUcnt]; 

			//count: how many instance of a type
			for(int r=0;r<numInst ;r++){
				if(colValues[r]==colUniqValue)					
					count++; 				
			}

//			System.out.println("colUniqValue "+colUniqValue+" "+count );

			probability = (double)count / numInst;
			if(count > 0) {
				entropy += (-probability * (Math.log(probability) / Math.log(2)));
			}
		}
		return entropy;
	}
	
	/*
	 * Calculate which attribute from unusedAttribues will produce the highest information gain
	 */
	public static int getHighestInfoGain(Matrix features, Matrix labels, ArrayList<Integer> unusedAttributes)	{
		
		double maxGain = -1;
		int bestAttribute = -1;
		int numberOfAttributes = features.cols();
//		for(int curAttributeColumn = 0; curAttributeColumn < unusedAttributes.size(); curAttributeColumn++)
		for(Integer curAttributeColumn : unusedAttributes)
		{
			double gain = calculateInfoGain(features, labels, curAttributeColumn);
			double entropy = calculateEntropyNominal(features.getColumn(curAttributeColumn), features.getUniqueValuesArray(curAttributeColumn));
//			System.out.println("\tInfo Gain at " + curAttributeColumn + ": " + gain);
//			System.out.println("\tEntropy at " + curAttributeColumn + ": " + entropy);
			if(gain > maxGain)	{
				maxGain = gain;
				bestAttribute = curAttributeColumn;
			}
		}
		
		return bestAttribute;
	}
	
	/*
	 * Calculate the info gain
	 */
	public static double calculateInfoGain(Matrix features, Matrix labels, int attributeColumnNum)	{
		
		double infoGain = 0.0;
		int valueCount = labels.valueCount(0);	//determines whether data is nominal or continuous
		double[] attributeColumnValues = features.getColumn(attributeColumnNum);
		double[] uniqueColumnValues = features.getUniqueValuesArray(attributeColumnNum);
		double attributeMedian = features.columnMedian(attributeColumnNum);
		double[] labelValues = labels.getColumn(0);
		double[] uniqueLabelValues = labels.getUniqueValuesArray(0);
		double classEntropy = calculateEntropyNominal(attributeColumnValues, uniqueLabelValues);
		
		if(valueCount == 0)	{	//if the label is continuous...
			infoGain = calculateInfoGainNumeric(attributeColumnValues, attributeMedian, classEntropy, labelValues, uniqueLabelValues);
		}
		else	{				//if the label is nominal...
			infoGain = calculateInfoGainNominal(attributeColumnValues, uniqueColumnValues, classEntropy, labelValues, uniqueLabelValues);
		}
		
		return infoGain;
	}
	
	/*
	 * Calculates information gain for nominal data sets - was called calculateInfoGain
	 */
	public static double calculateInfoGainNominal (double colOfAttrValues[], double attrUniqVals[], double classEntropy, double[] labelValues, double[] uniqueLabelValues){
		double entropy = 0;
		double subEntropy = 0;
		double probability = 0;
		int attrValCnt =0;
		double attrEntropy =0;
		double infoGain=0;

		ArrayList<Double> subEntropies = new ArrayList<Double>();

		int numInst= colOfAttrValues.length;

		if(colOfAttrValues.length == 0) {
			return 0; //no instances to consider
		}

		//loop for each of the distinct value of the attribute
		for(int colValUcnt=0;  colValUcnt<attrUniqVals.length ; colValUcnt++) {

			entropy =0;
			subEntropy = 0;
			attrValCnt =0;			
			double colUniqValue = attrUniqVals[colValUcnt]; 
			int attrClCnt;
			HashMap<Double, Integer> attrValClMap = new HashMap<Double, Integer>();

//			System.out.println("attribute value: "+colUniqValue);

			//Keeps track of how the instances are distributed across different classes
			//Initialization
			for(int label=0;label<uniqueLabelValues.length;label++)
				attrValClMap.put(uniqueLabelValues[label], 0);

			for(int r=0;r<numInst ;r++){	
				if(colOfAttrValues[r]==colUniqValue)	{		
					attrClCnt = attrValClMap.get(labelValues[r]);
					attrClCnt++;
					attrValClMap.put(labelValues[r], attrClCnt);
					attrValCnt++; 	
				}
			}

			if(attrValCnt > 0) { //if greater than 0 like 8 in my example
				for(int cl=0;cl<uniqueLabelValues.length;cl++){					
					int clattrCnt = attrValClMap.get(uniqueLabelValues[cl]);
					if(clattrCnt>0){ //If there are items in a particular class
						probability = (double)clattrCnt / attrValCnt;
						entropy += (-probability * (Math.log(probability) / Math.log(2)));
					}
				}				
//				System.out.println("attrValCnt  "+ attrValCnt+" numInst  "+ numInst);
				subEntropy = ((double)attrValCnt/numInst)*entropy;
//				System.out.println("entropy: "+entropy+" subEntropy of attr:  "+ subEntropy);
			}		
			subEntropies.add(subEntropy);
		}	//for each distinct value of entropy

		for(double d : subEntropies)
			attrEntropy += d; //This is E(Age) in Dr. Ng's Lecture of Data Mining

		//Info gain= clEntropy- sum of all sub entropies
		infoGain = classEntropy-attrEntropy;  //This is Gain(Age) in Dr. Ng's Lecture of Data Mining

		System.out.println("Info gain: "+	infoGain);
		return infoGain;
	}

	/*
	 * Calculates information gain for numeric data sets - was once called calIGainForNumeric
	 */
	public static double calculateInfoGainNumeric (double colValues[], double medOfColVals, double classEntropy, double[] labelValues, double[] uniqueLabelValues){

		int attrClCnt = 0;
		double entropy = 0;
		double subEntropy = 0;
		double infoGain=0;
		
		double probability=0;		

		//Currently for simplicity considering just two unique values for attribute
		//all instances less than has attribute value less than the median value are lower bound 
		int lowBoundInstances =0;
		//all instances less than has attribute value more than or=the median value are upper bound
		int upBoundInstances =0;

		int numInst= colValues.length;
		if(colValues.length == 0) {
			return 0; //no instances to consider
		}

		HashMap<Double, Integer> attrVal1ClMap = new HashMap<Double, Integer>();
		HashMap<Double, Integer> attrVal2ClMap = new HashMap<Double, Integer>();
		//Keeps track of how the instances are distributed across different classes
		//Initialization
		for(int label=0;label<uniqueLabelValues.length;label++){
			attrVal1ClMap.put(uniqueLabelValues[label], 0);
			attrVal2ClMap.put(uniqueLabelValues[label], 0);
		}		

		for(int r=0;r<numInst ;r++){
			//attribute values that belong to lower bound of the attribute
			if(colValues[r]<medOfColVals){
				attrClCnt = attrVal1ClMap.get(labelValues[r]);
				attrClCnt++;
				attrVal1ClMap.put(labelValues[r], attrClCnt);
				lowBoundInstances++;
			}
			else { //attribute values that belong to upper bound of the attribute
				attrClCnt = attrVal2ClMap.get(labelValues[r]);
				attrClCnt++;
				attrVal2ClMap.put(labelValues[r], attrClCnt);
				upBoundInstances++;
			}
		}

//		System.out.println("lowBoundInstances cnt: "+lowBoundInstances +"  upBoundInstances cnt: "+upBoundInstances);

		if(lowBoundInstances > 0) {
			entropy=0;
			for(int cl=0;cl<uniqueLabelValues.length;cl++){					
				int clattrCnt = attrVal1ClMap.get(uniqueLabelValues[cl]);
				if(clattrCnt>0){ //If there are items in a particular class
					probability = (double)clattrCnt / lowBoundInstances;
					entropy += (-probability * (Math.log(probability) / Math.log(2)));
				}
			}			
			subEntropy = ((double)lowBoundInstances/numInst)*entropy;			
		}
		if(upBoundInstances > 0) {
			entropy=0;
			for(int cl=0;cl<uniqueLabelValues.length;cl++){					
				int clattrCnt = attrVal2ClMap.get(uniqueLabelValues[cl]);
				if(clattrCnt>0){ //If there are items in a particular class
					probability = (double)clattrCnt / upBoundInstances;
					entropy += (-probability * (Math.log(probability) / Math.log(2)));
				}
			}			
			subEntropy += ((double)upBoundInstances/numInst)*entropy;			
		}
		
		infoGain=classEntropy - subEntropy;
		
		return infoGain;
	}
}
