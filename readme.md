**Bayes Decision Rule Classifier**<br /> 

Bayes theorem is used for find conditional probability of a hypothesis based on its prior probability.for an given sample in bayes theorem best estimate for the parameter is the values gives thw maximum probability for the outcome.<br /> 

**Explanation of the Question 1:**<br /> 
we have to generate 2-dimensional data points that are distributed according to the Gaussian distribution where the mean is given as m = [0, 0]  transpose and  covariance matrix S as 2*2 matrices that are <br /> 
s1=[[0.2,0],[0,2]]<br /> 
s2=[[2,0],[0,0.2]]<br /> 
s3=[[1,0.5],[0.5,1]]<br /> 
s4=[[0.3,0.5],[0.5,2]]<br /> 
s5=[[0.3,-0.5],[-0.5,2]]<br /> 
for execution the code libaries to be imported are numpy, random and matplotlib.<br /> 
In the question N was given as 500, so for ploting the distribution . x and y are taken as two variable where np.random.multivariate_normal was declare for calculating mean covarience for 500 data, and then transpose was done.<br /> 
By applying X,Y=np.random.multivariate_normal(m,s,500).T  calcution was made for s1,s2,s3,s4,s5 and graph for X,Y was plot to generated numbers for visualize the distributions<br /> 

**Explanation of the Question 2:**<br /> 
we have to generate 2-dimensional data points that are distributed according to the Gaussian distribution where the mean is given as<br /> 
 m1=[0,0]Transpose<br /> 
m2=[1,2]transpose<br /> 
and the covarience matrix is same for s1 and s2 so the variable for covarience matrix is taken as s<br /> 
s=[[0.8,0.2],[0.2,0.8]]. <br /> 
After initializing s and m1, m2 two variable was taken X which consist of 1000 data and Y which consist of 5000 data. X has been taken in two part X1 and X2 samples were divided into two equal half for training and testing.<br /> 
same thing has been done for Y.<br /> 
By applying X,Y=np.random.multivariate_normal(m,s,sample).T calculation was made for X1, X2, Y1, Y2.<br /> 
A in the question it was given that X is traning data and Y is testing Data, training and testing will be done on the basis of  he squared Euclidean distance-based classifier.<br /> 
Before solving Euclidean distance first we have to find centroid of training data using the formula<br /> 
centroid_X1=np.mean(X1, axis=1)<br /> 
print(X1)<br /> 
centroid_X2=np.mean(X2, axis=1)<br /> 
print(X2)<br /> 

After calculating the centroid euclidean distance was define as<br /> 
def euclidean_distance(p,q):<br /> 
    distance=0<br /> 
    p=np.asarray(p)<br /> 
    q=np.asarray(q)<br /> 
    
    for i in range(2):
        distance = distance + (np.square(p[i] - q[i])) 
                                   
    return np.sqrt(distance)

After defining euclidean distance classification error was calculated for Y1 and Y2. In the output we can see that classification error for Y1 is 0.0 and classification error for Y2 is 0.0008 and total classification error is 0.0008<br /> 

**Explanation of the Question 3:**<br /> 
For question number 3 math library was imported to find the value of pi
Initially it is a 5 dimensional  data vectors, and we have to use gaussian distribution where mean are m1 = [0,0,0,0,0]T  and m2 = [1,1,1,1,1] T and the covariance matrices are S1 = [[0.8, 0.2, 0.1, 0.05, 0.01],[0.2, 0.7, 0.1, 0.03, 0.02],[0.1, 0.1, 0.8, 0.02, 0.01], [0.05, 0.03, 0.02, 0.9, 0.01], [0.01, 0.02, 0.01, 0.01, 0.8]]
S2 = [[0.9, 0.1, 0.05, 0.02, 0.01],[0.1, 0.8, 0.1, 0.02, 0.02], [0.05, 0.1, 0.7, 0.02, 0.01], [0.02, 0.02, 0.02, 0.6, 0.02], [0.0.1, 0.02, 0.01, 0.02, 0.7]] at first we have to find for 50 sample 
using X=np.random.multivariate_normal(m,s,sample).T for X1 and X2 after that again mean and covariance was calculated for find x1 and x2 where samples are 10,000 training and testing data is define as 
x = np.random.multivariate_normal(mean,S,sample) .T for x1,x2 sample is taken as 5000 and 5000.
after that navie bayes classifier is define as p *= (1/(2*(math.pi)*(cov[j]))**(1/2))*(np.exp(-((x[j] - m[j])**2)/(2*cov[j])))
As there are two variable p1 and p2 for both the variable error probability was calculated and total error probability was measured.
At the end bar graph was plot to show the result analysis.
 
