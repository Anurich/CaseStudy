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
  8. [alt text](http://url/to/img.png)
  
