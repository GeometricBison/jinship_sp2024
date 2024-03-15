from torch import cuda
import transformers
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('english')
stopwords.append('the')
stopwords.append('and')
stopwords.append('to')
stopwords.append('of')
stopwords.append('in')
stopwords.append('with')
stopwords.append('an')
stopwords.append('The')
stopwords.append('And')
stopwords.append('To')
stopwords.append('Of')
stopwords.append('In')
stopwords.append('With')
stopwords.append('An')
stopwords.append('A')
stopwords.append('As')
stopwords.append('as')
stopwords.append('it')
stopwords.append('It')


def generate_topic_name(model_generator, list_of_words):

    words = ', '.join([i for i in list_of_words])

    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics.
    <</SYS>>
    """

    example_prompt = """
    I have a topic that is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

    [/INST] Environmental impacts of eating meat
    """

    main_prompt = """
    [INST]
    I have a topic that is described by the following keywords: '[{}]'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
    [/INST]
    """.format(words)

    prompt = system_prompt + example_prompt + main_prompt

    response = model_generator(prompt)

    if prompt in response[0]["generated_text"]:
        response = response[0]["generated_text"].replace(prompt, "")

    return response


def remove_stopwords(text: str):
    output= ' '.join([i for i in text.split() if i not in stopwords])
    return output



model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

dataset = load_dataset("valurank/Topic_Classification", split="train")

# Extract abstracts to train on and corresponding titles
wiki_data = dataset["article_text"]
wiki_data = [text for text in wiki_data if type(text) == str]

# print(type(wiki_data))

processed_data = [remove_stopwords(datum) for datum in wiki_data]


topic_model = BERTopic(
  # Hyperparameters
  nr_topics=30,
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(processed_data)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto'
)

generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=10000,
    repetition_penalty=1.1
)


all_topics = topic_model.get_topic_info()
for i in range(len(all_topics)):
    words = all_topics["Representation"][i]
    documents = all_topics["Representative_Docs"][i]
    all_topics["Name"][i] = generate_topic_name(generator, words)

print(all_topics["Name"])