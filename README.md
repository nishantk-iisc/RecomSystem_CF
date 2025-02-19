# Amazon Apparel Recommendation System

Our goal is to recommend similar apparel items to e-commerce users, leveraging text and image features.

## Business Objective
We aim to enhance the user experience by recommending relevant apparel items. Notably, around 35% of Amazon's revenue comes from product recommendations.

## Approach

We employ two main strategies:

1. Content-based recommendation:
   - This method relies on item attributes such as title text, description text, and images to recommend similar products.
   
2. Collaborative filtering-based recommendation:
   - This approach focuses on user behaviour, where recommendations are made based on the actions and preferences of similar users.

If:
    - User U1 has purchased or searched for items I1, I2, I3
    - User U2 has purchased or searched for items I1, I3, I4

Then, we recommend to User U3 item I3, as User U3 has shown interest in item I1, which is also interesting to similar users. This approach, known as collaborative filtering, leverages user behaviour to make recommendations.

However, due to Amazon's privacy policies, user data is inaccessible. Therefore, we focus on content-based recommendation methods instead of collaborative filtering.


## Plan of Attack

Here's systematic approach for this project:

1. **Data Acquisition:** obtain the necessary data.
2. **Data Cleaning:** The acquired data undergoes cleaning to ensure quality and consistency.
3. **Text Processing (NLP):** We apply Natural Language Processing techniques to process and analyze text data.

Then, we address the problem using both Text-based and Image-based methods.

4. **Text-based Product Recommendation:**
   - **Bag of Words:** A simple technique to represent text data.
   - **Term Frequency-Inverse Document Frequency (TF-IDF):** Assigns weights to words based on their frequency in documents.
   - **Word2Vec:**
     - **Avg W2V:** Computes the average vector of word embeddings.
     - **TF-IDF weighted W2V:** Uses TF-IDF weights to create weighted word vectors.

5. **Image-based Product Recommendation:**
   - Utilizing Convolutional Neural Networks (CNN) for deep learning-based image analysis.

6. **A/B Testing:** This step involves testing and comparing different models or approaches to determine the most effective one.


### 1. Data Acquisition:

I used Amazon's Product Advertising API to gather the necessary data for data acquisition. Specifically, we focused on women's tops as our dataset, which consisted of 183,000 data points, each with 19 features. These features include:

1. **asin:** Amazon Standard Identification Number.
2. **author:** The product's author (if applicable).
3. **availability:** Availability status of the product.
4. **availability_type:** Type of availability (e.g., In Stock, Out of Stock).
5. **brand:** The brand to which the product belongs.
6. **color:** Color information of the apparel item. This could include multiple colours such as "red and black stripes."
7. **editorial_review:** Editorial review of the product.
8. **formatted_price:** The product price in a formatted manner.
9. **large_image_url:** URL of the large image associated with the product.
10. **manufacturer:** The manufacturer of the product.
11. **medium_image_url:** URL of the medium-sized image associated with the product.
12. **model:** The model number of the product.
13. **product_type_name:** The type of apparel (e.g., SHIRT, TSHIRT).
14. **publisher:** The publisher of the product.
15. **reviews:** Reviews or ratings associated with the product.
16. **sku:** Stock Keeping Unit, a unique identifier for the product.
17. **small_image_url:** URL of the small-sized image associated with the product.
18. **title:** The title or name of the product.

However, for our project, we focused on utilizing the following 6 features primarily:

1. **asin:** Amazon Standard Identification Number.
2. **brand:** The brand of the product.
3. **color:** Color information of the apparel item.
4. **product_type_name:** Type of apparel (e.g., SHIRT, TSHIRT).
5. **medium_image_url:** URL of the medium-sized image associated with the product.
6. **title:** The title or name of the product.
7. **formatted_price:** The product price in a formatted manner.

These features were deemed most relevant for our apparel recommendation system.

### 2. Data Cleaning

#### Basic Stats for Each Feature
During the data cleaning process, we conducted an analysis for each feature in the dataset:

- Checked for missing values in each feature to determine the extent of data completeness.
- Identified the most frequent item in each feature and calculated the number of times it appeared in the dataset.
- Identified the dataset's top 10 most frequently occurring words.

For data processing efficiency, we decided to remove all data points that had missing values in any of the features. Specifically, we removed data points with null values in the **formatted_price** and **color** columns. After this cleaning step, we were left with approximately 28,000 data points.

Additionally, we utilized the **medium_image_url** feature, which contains URLs for medium-sized images of apparel items. We retrieved and saved all these images using the Python Imaging Library (PIL) for further analysis and processing.

#### Removal of Near-Duplicate Items
One common issue observed in the dataset was the presence of near-duplicate items, especially in the **title** feature. These near-duplicates often varied only in colours or sizes within the product titles. For example:

```
25. tokidoki The Queen of Diamonds Women's Shirt X-Large
26. tokidoki The Queen of Diamonds Women's Shirt Small
27. tokidoki The Queen of Diamonds Women's Shirt Large

75004. EVALY Women's Cool University Of UTAH 3/4 Sleeve Raglan Tee
109225. EVALY Women's Unique University Of UTAH 3/4 Sleeve Raglan Tees
120832. EVALY Women's New University Of UTAH 3/4-Sleeve Raglan Tshirt
```

I implemented a process to remove these near-duplicate items to address this issue. We achieved this by comparing the words within the titles and identifying titles that were very similar but varied primarily in colour or size information. Additionally, we removed titles that contained fewer than 5 words, as they were often less descriptive and informative.

After completing these cleaning steps, our dataset was refined to around 16,000 data points, ensuring a more accurate and focused dataset for our recommendation system.


### 3. Text Preprocessing

In this step, basic text preprocessing tasks are performed to clean and prepare the textual data for further analysis. These preprocessing steps typically include:

- Removing stopwords: Stopwords are common words like "the," "is," "and," etc., which are often irrelevant for analysis and can be removed.
- Removing spaces and alphanumeric characters: We clean the text by removing extra spaces, punctuation marks, and non-alphanumeric characters.
- Lowercasing: Converting all text to lowercase ensures consistency and reduces the complexity of text analysis.

### 4. Text-Based Product Recommendation

I primarily utilize our dataset's **title** feature for text-based product recommendations. This approach involves converting the textual data into numerical vectors, which can be used to find similarities between different items. We explore several techniques for text vectorization:

1. **Bag of Words (BoW):** This method represents text as a sparse vector where each dimension corresponds to a unique word in the corpus. The value in each dimension indicates the frequency of that word in the text.
2. **Term Frequency-Inverse Document Frequency (TF-IDF):** TF-IDF calculates the importance of a word in a document relative to its frequency in the entire corpus. It penalizes common words and gives more weight to rare, informative words.
3. **Word2Vec:** Word2Vec is a deep learning technique representing words as dense vectors in a continuous vector space. It captures semantic relationships between words and can preserve contextual information.

Within the Word2Vec approach, explore two variations:
- **Average Word2Vec (Avg W2V):** This method calculates the average vector representation of all words in a text.
- **TF-IDF Weighted Word2Vec:** I use TF-IDF scores to weight each word's vector before calculating the average vector representation.

The core idea behind the text-based recommendation is to convert textual information into meaningful numerical representations that capture the essence and context of the data. These vector representations allow us to compute item similarities based on their textual descriptions.

For a detailed explanation of these concepts and their implementation, refer to the [kaggle kernel](https://www.kaggle.com/shashanksai/text-preprocessing-using-python).

### 5. Image-Based Product Recommendation

For image-based product recommendation, I employ transfer learning using the VGG16 algorithm. This involves leveraging pre-trained models from VGG16, which are trained on extensive image datasets. We can convert images into meaningful numerical vectors that capture visual features using these pre-trained models.

The results and outputs of this approach can be found in the accompanying Jupyter Notebook (Ipynb) file.

### 6. A/B Testing Overview

A/B testing is a performance metric used to evaluate models in production. It involves dividing users into two groups, A and B. Both groups are exposed to different versions of a model or algorithm. The performance metrics of both versions are then compared to determine which one yields better results. This process helps make informed decisions about deploying the most effective model or algorithm.

The Results/Output section showcases the query image and the recommended apparel as follows:

**Query Image**
![Query Image](images/q.jpg)

**Recommended Apparel**
1. ![Image 1](images/1.jpg)
2. ![Image 2](images/2.jpg)
3. ![Image 3](images/3.jpg)
4. ![Image 4](images/4.jpg)
5. ![Image 5](images/5.jpg)
6. ![Image 6](images/6.jpg)
7. ![Image 7](images/7.jpg)
8. ![Image 8](images/8.jpg)
9. ![Image 9](images/9.jpg)
