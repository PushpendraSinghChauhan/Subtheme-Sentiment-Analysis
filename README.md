# Subtheme-Sentiment-Analysis
Here, our task is to develop an approach that given a sample will identify the subthemes along with their respective sentiments. 
#### For Example :
   INPUT ---> “One tyre went missing, so there was a delay to get the two tyres fitted. The way garage dealt with it was fantastic.” <br>
   OUTPUT --> ['incorrect tyres sent negative', 'garage service positive', 'wait time negative']  <br>

## My Approach :    
--> Read the csv file named "Evaluatoion-Dataset.csv" using pandas library. <br>
--> Then, I performed some data-preprocessing such as removing duplicate rows, replacing NaN values with 'no_label', combined all the labels into a single target variable named "Labels" and then removing 'no_label' from them etc. <br> 
--> Then, we perform some data visualizations to check how classes are distributed. This helps us to remove some datapoints which are like noisy points and also help us in dealing with the problem of imbalanced classes to some extent. <br> 
--> I have used Bart LLM model developed by facebook. I just used it but I haven't fine-tuned it because it has lot of parameters so fine-tuning is not possible with the cpu and also not using Google Colab because they give GPUs to use for a very limited time. <br>
#### Results of Bart Model : <br>
Text = “One tyre went missing, so there was a delay to get the two tyres fitted. The way garage dealt with it was fantastic.” <br>
Top 5 Labels for Bart model : ['garage service positive','facilities positive', 'response time positive', 'balancing positive', 'advisor/agent service positive'] <br> 
--> I have fine-tuned only 1 Bert model named "Bert-base-uncased" and not "Bert-large-uncased" because "Bert-base-uncased" have 110 million parameters while "Bert-large-uncased" have 340 million parameters. Training "Bert-base-uncased" model on Google Colab took approximately 2 hours on a GPU. <br>
--> I have used "MultiLabelBinarizer" to convert our labels into vectors and then at the inference time we have used reverse_transform() to convert our vector back to class labels. <br>
--> Parameters Used for fine-tuning : <br>
Max_token_length = 150 (we have got it by plotting PDF and CDF plot of token length of reviews/datapoints.) <br>
EPOCHS : 10 <br> 
Batch_size : 16 <br> 
learning_rate : 2e-5 <br> 
loss : CrossEntropy <br>
num_labels : 55 (After data preprocessing, we left with 55 unique subtheme labels) <br> 

## Results : 

Epoch 1/10 <br>
----------  <br>
Train loss 0.09334013338391141  accuracy 0.9898305084745763  <br>
Validation loss 0.026192670153892216  accuracy 0.9956869993838571  <br>

Epoch 2/10  <br>
----------   <br>
Train loss 0.03262386077142142  accuracy 0.9938366718027735   <br>
Validation loss 0.022378485071171513  accuracy 0.9956869993838571  <br>

Epoch 3/10  <br>
----------   <br>
Train loss 0.02408523136650079  accuracy 0.9938366718027735  <br>
Validation loss 0.024408606408328256  accuracy 0.9944547134935305  <br>

Epoch 4/10  <br>
----------  <br>
Train loss 0.012272305978006761 accuracy 0.9959938366718027  <br>
Validation loss 0.03410159281338565 accuracy 0.9950708564386938  <br>

Epoch 5/10  <br>
----------  <br> 
Train loss 0.007397184166782008 accuracy 0.99768875192604  <br>
Validation loss 0.037750929538970905 accuracy 0.9950708564386938  <br>

Epoch 6/10  <br>
----------   <br>
Train loss 0.004845062871116645 accuracy 0.9989214175654854  <br>
Validation loss 0.038309747617698606 accuracy 0.9944547134935305  <br>

Epoch 7/10 <br>
---------- <br>
Train loss 0.002799637230019099 accuracy 0.9993836671802774 <br>
Validation loss 0.03877896219708811 accuracy 0.9944547134935305 <br>

Epoch 8/10 <br>
----------  <br>
Train loss 0.0008856927881532307 accuracy 0.9998459167950693  <br>
Validation loss 0.041500866751519816 accuracy 0.9944547134935305 <br>

Epoch 9/10 <br>
---------- <br>
Train loss 0.0006891316638197383 accuracy 0.9998459167950693 <br>
Validation loss 0.0430650869100503 accuracy 0.9944547134935305 <br>

Epoch 10/10 <br>
----------  <br>
Train loss 0.0005153585583048419 accuracy 1.0  <br>
Validation loss 0.043511063997392935 accuracy 0.9944547134935305 <br>

#### Some Plots :

![Screenshot (268)](https://github.com/PushpendraSinghChauhan/Subtheme-Sentiment-Analysis/assets/34591830/c3110451-3d66-422d-8bc4-697e5d08a46f)

![Screenshot (269)](https://github.com/PushpendraSinghChauhan/Subtheme-Sentiment-Analysis/assets/34591830/645032fd-2867-479f-8f42-0104cd7d4a6e)

![Screenshot (270)](https://github.com/PushpendraSinghChauhan/Subtheme-Sentiment-Analysis/assets/34591830/62421ae7-effd-4d60-9d9b-22ac750c2b61)

#### Conclusion :
From above we can clearly see that model starts overfitting after 4 epochs <br> 

## Limitations And Approach For Future Improvements :
--> More task in data-preprocessing : Removal of stopwords, removal of special symbols (!,@,# etc.), convert all the strinngs to lowercase etc. <br>
--> Can try various values of learning rate, batch-size, dropout rate, loss  <br> 
--> Can do undersampling/oversampling in order to address the problem of overfitting. <br> 
--> Can also use larger models like "Bert-large-uncased",Llama 2, Bart etc for fine-tuning but it will take more computational power of GPU because they contain parameters in billions. <br>
--> Can also save models everytime when validation loss decreases from previous epochs in order to get best fine-tunes model <br>
--> Since our model is overfitting so we can use regularization like ridge (L2 regularization), lasso (L1) and also we can use batch normalization to avoid the problem of internal covariate shift because our model is deaper as "Bert-base-uncased" model has 12 hidden layers.  <br> 

## Inference : 
INPUT --> “One tyre went missing, so there was a delay to get the two tyres fitted. The way garage dealt with it was fantastic.” <br> 
Top 3 Labels for Fine-Tuned model: [('refund positive', 'wait time negative', 'wait time positive')] 

## Libraries to install to run this notebook :
--> numpy, pandas, matplotlib, seaborn, tensorflow, torch,  transformers, scikit-learn 
