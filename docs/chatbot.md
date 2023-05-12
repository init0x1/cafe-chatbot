# ChatBot(Cafe-Chatbot)



### Project members : 
1. Abdelrahman Ali Abdelhafeez (2001470)
2. Kareem Mohamed Ramdan (2001952)
3. Mohamed yasser hussein  (2001886)
4. Zaid Ali Abdullah (2001572)
5. Mohamed Ahmed Ali (2001632)

## Project description
Cafe Chatbot is a simple chatbot that can help customers with their queries related to cafes. It uses a dataset of questions and answers to provide responses to user inputs. The chatbot is developed using Python and uses Natural Language Processing (NLP) Algorithms to generate responses.

## Project goal
The goal of this project is to provide a conversational interface for customers to get information about the cafe, such as the menu, hours of operation, and location. The chatbot should be able to handle a variety of user queries and provide accurate and relevant responses in a timely manner.

## Machine learning algorithms used in project

 The Cafe-Chatbot project uses a combination of machine learning and `Rule & Retrieval-based`  approaches to generate responses to user queries. 

 Two machine learning algorithms have been used in this project: `Naive Bayes` and `Support Vector Machines (SVMs).`


## Dataset

The dataset used by the chatbot is stored in the `cafedataset.csv` file. The dataset contains questions and answers related to cafes. The file is loaded using the Pandas library and processed to remove any `NaN values`.

dataset source : https://www.kaggle.com/datasets/sonalibhoir/cafe-chatbot-dataset?select=conversationo.csv

## Naive Bayes

The Naive Bayes algorithm is a probabilistic machine learning algorithm commonly used for text classification tasks. It calculates the probability of a given input belonging to a particular class by multiplying the probabilities of each of its features given that class.

 The algorithm is implemented using the `scikit-learn` library in Python.

1. The dataset used in the project is loaded from a CSV file using the `pandas` library. The `TfidfVectorizer` class from the scikit-learn library is used to `convert the questions in the dataset into feature vectors`, which are then used to train the `Multinomial Naive Bayes classifier.`

2. A function called `'nb_response'` is defined to generate responses using the trained Naive Bayes classifier. This function takes user input as an argument, converts it into a feature vector using the vectorizer, and predicts the label for the feature vector using the trained Naive Bayes classifier.

3. The chatbot is tested using a loop that iterates through the questions in the dataset and generates responses using the `'nb_response'` function. The accuracy of the chatbot is calculated by comparing the predicted labels to the expected labels in the dataset.

4. Finally, the chatbot is run using a while loop that takes user input and generates responses using the 'nb_response' function until the user types "quit".

### Accuracy: 45.04%

## Rule-based aproache
A rule-based chatbot is a type of chatbot that uses a set of predefined rules to generate responses to user inputs. These rules are typically created manually and cover a wide range of possible user inputs and responses.

1. Load the dataset: The chatbot starts by reading a CSV file containing a list of questions and answers that it will use to generate responses.

2. Create a dictionary of rules: The chatbot uses the dataset to create a dictionary of rules based on the questions and their corresponding answers. Each question in the dataset is treated as a key in the dictionary, and the corresponding answers are stored as a list of possible responses.

3. Define the chatbot response function: A function is defined to handle user input and generate responses. The function takes the user's input as an argument, searches through the rules dictionary for a matching question, and returns a randomly chosen response from the list of possible answers for that question. If no matching question is found, the chatbot returns a default "I'm sorry, I don't understand your question" response.

4. Test the chatbot: The chatbot is tested by iterating through the questions in the dataset and comparing the chatbot's responses to the expected answers. The accuracy of the chatbot is calculated by dividing the number of correct responses by the total number of responses.

5. Run the chatbot: Finally, the chatbot is run in a loop that continually prompts the user for input and generates responses using the chatbot response function until the user terminates the program.

### Accuracy: 28.39%

## Support Vector Machines (SVMs).

Support Vector Machines (SVMs) is a popular machine learning algorithm used for classification and regression analysis. In classification, SVMs are used to separate data into different categories by finding the best hyperplane that can separate the classes. The SVM algorithm seeks to find the hyperplane that maximizes the margin between the two classes of data points. 

1. Load the dataset: The code reads in the data from a CSV file using pandas and removes any NaN values from the DataFrame.

2. Create feature vectors: The code creates feature vectors for the questions in the dataset using the TfidfVectorizer class from the scikit-learn library. This converts the text data into numerical vectors that the SVM algorithm can use to make predictions.

3. Train the SVM: The code uses the SVM algorithm to train a classifier on the feature vectors and the corresponding labels.

4. Generate responses: The code defines a function called 'svm_response' that takes user input as an argument, converts it into a feature vector using the vectorizer, and predicts the label for the feature vector using the trained SVM classifier.

5. Test the chatbot: The code tests the accuracy of the chatbot by iterating through the questions in the dataset and comparing the predicted labels to the expected labels.
6. run the chatbot

### Accuracy: 69.48%

## Retrieval-Based approache

The retrieval-based approach is a type of chatbot that retrieves answers from a pre-existing database of responses, rather than generating new responses. Here is a step-by-step explanation of the code:

1. The first step is to import the necessary libraries and load the dataset from a CSV file using pandas.

2. The dataset is then pre-processed by removing any rows with NaN values in the "Question" and "answer" columns.

3. An instance of the TfidfVectorizer class is created to convert the text data into a numerical format that can be processed by the algorithm. This involves creating a vocabulary of all the unique words in the dataset and assigning a weight to each word based on its frequency in each document (i.e., each question).

4. The vectorizer is then fit to the dataset to create the vocabulary and assign weights to the words.

5. A function is defined to handle user input and return a response. When the user inputs a question, the function converts the question into a feature vector using the TfidfVectorizer object, and then calculates the cosine similarity between the user input vector and the question vectors in the dataset. The question-answer pair with the highest similarity score is selected as the response.

6. The chatbot is tested by looping through each question-answer pair in the dataset and comparing the chatbot's response to the expected response. The accuracy of the chatbot is calculated as the percentage of correct responses.

7. Finally, the chatbot is run in an interactive loop that allows the user to input questions and receive responses. If the user inputs "quit", the loop is exited and the program terminates.

### Accuracy: 69.48%