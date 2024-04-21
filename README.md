# Case Study 
## Task --1 
* Under this task I need to make the therapist who will provide insight about, how to deal with depression, Anxiety, anger, etc.
* I mention about the approach I used to tackle this problem:
  1. First we need to find the emotion based on human speech or text.
  2. To calculate the emotion, I suggested we can do this using two approaches: One is to **fine-tuned** a SequenceClassificationModel, another is to use Prompt engineering. I coded both approaches. Which can be found under task 1; under **emotion_detection** directory which contain the code for SequencClassificationModel, on emotion dataset.
  3. But I used the prompt engineering approach to solve this problem.
  4. I am categorising two types of emotions: **angry, and depression**.
  5. I am using dataset regarding how to overcome depression, and how to perform anger management. So I am using these dataset for doing RAG for these two different type of emotions.
  6. The third category is otherwise if the llm suggested any emotion other than **angry and depressed**, I am simply using llm to generate the answer based on the question.
  7. I used langgraph to tackle this problem: My architecture is shown in the image below:
  ![alt text](https://github.com/Anurich/CaseStudy/blob/main/therapist_architecture.png)
  

## Task --2
* In this task I have fine-tuned two model, one is **information retreival model** and **summarization model**
* **Information Retreival Model:**   
  * The task is to find the way that for given query for specific domain, I will be able to find the corresponding response, or vice versa.
  * So I constructed the dataset that is preprocesed into format of [anchor, positive, negative]. Which is used to fine-tuned a sentence-transformers model. The loss function used in this is **MultipleNegativesRankingLoss**
  * Dataset used for this reddit-text.
  * Model used for this task is **"bert-base-uncased"**
  * Metrics used in this case is accuracy_manhattan, accuracy_euclidien, accuracy_cosinus.

* **Summarization**
  *  In case of summarization I have used the **samsum** dataset, Which contain the question and it's corresponding summarization.
  *  Model used to fine-tuned is facebook **"facebook/bart-large-cnn"**.
  *  I have used Peft with LORA configuration, to fine-tuned this model.
  *  Metrics used in this case is Rouge score.
 

## Library used:
1. Langchain
2. Langgraph
3. Transformers
4. Sentence-transformers
5. Pytorch

## Language
* Python==3.12.2
