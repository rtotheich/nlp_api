from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr


# load models for all tasks
# sentence similarity
model_similarity = BertForSequenceClassification.from_pretrained('./lilifinal_checkpoint/bert-base-uncased-finetuned')
tokenizer_similarity = BertTokenizer.from_pretrained('bert-base-uncased')
# emotion detection
model_emotion = AutoModelForSequenceClassification.from_pretrained('./rich_checkpoint-500')
tokenizer_emotion = AutoTokenizer.from_pretrained('./rich_checkpoint-500')
class_names_emotion = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
# named entity recognition
model_ner = AutoModelForTokenClassification.from_pretrained('./vijay_checkpoint-5268')
tokenizer_ner = AutoTokenizer.from_pretrained('./vijay_checkpoint-5268')


# define a function to compute similarity scores
def compute_similarity_score(source_sentence, compare_sentences):
  cosine_similarities_list = []
  for compare_sentence in compare_sentences:
    tokens_test = tokenizer_similarity(source_sentence, compare_sentence, return_tensors='pt', padding=True, truncation=True)
    output_test = model_similarity(**tokens_test)
    cosine_similarities_list.append(output_test.logits.tolist()[0])

  return cosine_similarities_list

# set page configuration
st.set_page_config(page_title="NLP Tasks", page_icon=":speech_balloon:", layout="wide")

# page for sentence similarity task
def sentence_similarity_page():
  st.title("Sentence Similarity Task")

  # Add input fields to the sidebar
  # with st.sidebar:
  source_sentence = st.text_input(label="Enter the source sentence:")
  compare_sentences=[]
  for i in range(3):
    compare_sentence = st.text_input(label=f"Enter sentence {i+1} to compare: ", key = f"compare_sentence_{i}")
    if compare_sentence:
      compare_sentences.append(compare_sentence)

  if st.button("Compute Similarity Scores"):
    if not source_sentence:
      st.error("Please enter a source sentence.")
    else:
      scores = compute_similarity_score(source_sentence, compare_sentences)
      # display results
      col1, col2 = st.columns([1, 2])
      with col1:
        st.write("Score")
        st.write("---")
        for i, score in enumerate(scores):
          print(i, score)
          st.write(f"{i+1}. {score[0]:.2f}")
      with col2:
        st.write("Sentence")
        st.write("---")
        st.write(f"Source Sentence: {source_sentence}")
        for i, compare_sentence in enumerate(compare_sentences):
          st.write(f"Compare Sentence {i+1}: {compare_sentence}")


# page for emotion detection task
def emotion_detection_page():
    st.title("Emotion Detection Task")
    
    source_sentence = st.text_input(label="Enter a sentence for emotion detection:")

    # for uploaded video
    uploaded_file = st.file_uploader("OR Upload a video file", type=["mp4", "mov"])
    if uploaded_file is not None:
      video_bytes = uploaded_file.read()
      video_name = uploaded_file.name

      # save the uploaded video file
      with open(video_name, "wb") as video_file:
        video_file.write(video_bytes)

      # extract audio from the video and convert it to text
      video = mp.VideoFileClip(video_name)
      audio = video.audio
      audio.write_audiofile("temp.wav")
      recognizer = sr.Recognizer()
      with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
      text_audio = recognizer.recognize_google(audio_data)


    # prediction
    def predict(text):
      with torch.no_grad():
        inputs = tokenizer_emotion(text, return_tensors="pt")
        output = model_emotion(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)[0].item()
        return (pred_label, class_names_emotion[pred_label])

    if st.button("Detect Emotion"):
      if not source_sentence and uploaded_file is None:
        st.error("Please enter a sentence or upload a file.")
      else:
        output_sentence = ''
        if uploaded_file is not None:
          emotion = predict(text_audio)
          output_sentence = text_audio
        else:
          emotion = predict(source_sentence)
          output_sentence = source_sentence        
        # display results
        col1, col2 = st.columns([1, 2])
        with col1:
          st.write("Predicted Emotion")
          st.write("---")
          # print(emotion[1])
          # display the predicted label and its corresponding class name
          st.write(f"{emotion[0]}. {emotion[1]}")
        with col2:
          st.write("Sentence")
          st.write("---")
          st.write(f"Source text: {output_sentence}")


# Page for Named Entity Recognition Task
def named_entity_recognition_page():
  st.title("Named Entity Recognition Task")
  
  source_sentence = st.text_input(label="Enter a sentence for named entity recognition:")
  source_entity = st.text_input(label="Enter an entity for named entity recognition (ORG, PER, LOC, MISC):")

  # prediction
  def predict(text:str):
    inputs = tokenizer_ner(text, return_tensors="pt")
    ids_list = inputs['input_ids']
    tokens = inputs.tokens()
    with torch.no_grad():
      logits = model_ner(**inputs).logits
      predictions = torch.argmax(logits, dim=2)
      predicted_token_class = [model_ner.config.id2label[t.item()] for t in predictions[0]]
    return (tokens, ids_list, predicted_token_class)


  #function to handle any type of entity
  def get_entities(text, entity):
    labels_list = predict(text)[2]

    tokens_list = text.split()
    entities_list = list()
    pairings_list = tuple(zip(tokens_list, labels_list))

    index = 0
    length_of_pairings_list = len(pairings_list)

    while index < length_of_pairings_list:
      current_index = index
      new_entity_name = ''
      while current_index < length_of_pairings_list and entity in pairings_list[current_index][1]:
        new_entity_name += pairings_list[current_index - 1][0] + ' '
        current_index += 1

      if len(new_entity_name) > 0:
        new_entity_name = new_entity_name.strip()
        entities_list.append(new_entity_name)

      index += 1 if index == current_index else current_index

    return entities_list if len(entities_list) > 0 else 'No entities found'


  if st.button("Recognize the Named Entity"):
    if not source_sentence:
      st.error("Please enter a sentence.")
    elif not source_entity:
      st.error("Please enter an entity.")
    else:
      ner = get_entities(source_sentence, source_entity)        
      # display results
      col1, col2 = st.columns([1, 2])
      with col1:
        st.write("Recognized Entity")
        st.write("---")
        st.write(f"{ner}")
      with col2:
        st.write("Sentence")
        st.write("---")
        st.write(f"Source Sentence: {source_sentence}, Source Entity: {source_entity}")


# main app
pages = {
  "Sentence Similarity": sentence_similarity_page,
  "Emotion Detection": emotion_detection_page,
  "Name Entity Recognition": named_entity_recognition_page
}

selection = st.sidebar.radio("Go to task", list(pages.keys()))
pages[selection]()