# Langchain + OpenAI + CVS + FAISS Vector DB

# Deploy and run the MovieBot App

Now that you have built your application, you can deploy it locally. Here’s how:

1. From your terminal window, `cd` Into `/llm-vectordb` folder where `main.py` is located
2. Download the dataset from https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movies.csv 
3. Run this command: `streamlit run main.py`
4. Open a browser window and navigate to `localhost:8501` to see your application running.
5. You can now chat with the bot and ask it questions about movies! For example:
   - Who starred in the movie Titanic?
   - What genre is The Matrix?
   - When was Jurassic Park released?

# How it works:

- Rotten Tomatoes movie data is loaded from the `.csv` file. 
- The loaded docs are indexed using Instructor-Large model and FAISS from Langchain package.
- User inputs are queried against this index using FAISS from Langchain. 
- The relevant text from the index is then sent to the OpenAI LLM endpoint using langchain. 
- Responses from the OpenAI Chat API are displayed in the Streamlit chat interface