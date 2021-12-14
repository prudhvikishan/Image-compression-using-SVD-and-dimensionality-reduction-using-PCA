# Image-compression-using-SVD-and-dimensionality-reduction-using-PCA

Goal of this academic project is to implement the Image compression using SVD and Dimensionality reduction of any dataset using PCA.

#Data should be placed in the data folder and all the generated outputs will also store in the data folder.
#Conversion from ascii-binary-ascii and Image compression using SVD and Error calculations are coded in Java.
#Java files used - SVD.java, Utils.java and DecomposeMatrix.java.
#SVD is our main java class which separates the given tasks.
#DecomposeMatrix class in DecomposeMatrix.java file decomposes the given matrix into U,S,VT and saves the header.txt and SVD.txt
#Utils.java file has the few methods like saveCompressedFileToBin, readCompressedBin, convertFloatToHalfPrecision.
#Note -- PGM file should not have comment line.
# jblas.jar is a matrix library for Java

Steps_of_Execution

Firstly step is to compile all the java files. This can be done by executing the command given below
----- javac -cp .:jblas.jar *.java

1st task: Convert ASCII image to binary image. This can be done by executing the command given below
----- java -cp .:jblas.jar SVD 1 CAS.pgm
program will use binary coding to save all necessary information in your asci pgm image. 
Used 2 bytes to save the wide of the image and 2 bytes to save the height of the image; 
used one byte for the grey scale levels; and use one byte for the grey level of each pixel.
Output saved as CAS_b.pgm 

2nd task: convert binary to ASCII image. This can be done by executing the command given below.
----- java -cp .:jblas.jar SVD 2 CAS_b.pgm
Output saved as CAS_copy.pgm.

3rd task: To decompose the matrix and getting image_header.txt and image_svd.txt. This can be done by executing the command given below
----- java -cp .:jblas.jar DecomposeMatrix CAS.pgm
Output saved as CAS_header.txt and CAS_SVD.txt

4th task: This task compress the image and then converts into binary file. This can be done by executing the command given below.
----- java -cp .:jblas.jar SVD 3 CAS_header.txt CAS_SVD.txt rank
The header.txt contains 3 integers (width, height, grey scale levels). SVD.txt contains 3 matrices.
Rank is 2.
Output saved as CAS_b.pgm.SVD. 

5th task: To recover the image. This can be done by executing the command given below.
----- java -cp .:jblas.jar SVD 4 CAS_b.pgm.SVD
Output saved as CAS_2.pgm

6th task: Error Calculations. This can be done by executing the command given below.
java -cp .:jblas.jar SVD 5 CAS.pgm CAS_2.pgm CAS_b.pgm.SVD

Outputs are THE RATE OF COMPRESSION AND MEAN SQUARE ERROR.

