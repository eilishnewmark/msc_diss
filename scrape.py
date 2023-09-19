import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict

# Function to fetch a random Wikipedia page title using Wikipedia API
def get_random_wikipedia_title():
    url = "https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=1&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "query" in data and "random" in data["query"] and len(data["query"]["random"]) > 0:
            return data["query"]["random"][0]["title"]
    print("Failed to fetch a random Wikipedia page title.")
    return None


# Function to fetch the Wikipedia page content
def get_wikipedia_page(title):
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {title} from Wikipedia.")
        return None

# Function to find instances of words in the given Wikipedia page
def find_instances_of_words(page_content, words):
    sentences = []
    if page_content is None:
        return sentences

    soup = BeautifulSoup(page_content, "html.parser")
    paragraphs = soup.find_all("p")

    found_words = []
    found_word_counts = defaultdict(lambda: 0)
    sentences_as_words = []
    for paragraph in paragraphs:
        for sentence in paragraph.text.split(".| "):
            sentence = sentence.rstrip().split(".")[:-1]
            sentences_as_words.extend(list(map(str.split, sentence)))
    for sentence in sentences_as_words:
        for word in words:
            if word.lower() in sentence:
                found_words.append(word.lower())
                found_word_counts[word.lower()] += 1
                sentences.append(" ".join(sentence).strip() + ".")
                break
    
    return sentences, found_words, found_word_counts

# Main function to scrape 1000 random Wikipedia pages and find sentences containing the predetermined words
def main():
    # Pre-determined list of words to search for
    predetermined_words = ['upgrade', 'rethink', 'redo', 'misprint', 'beneficiate', 'coagulate', 'exfoliate', 'hyphenate']
    # predetermined_words = ["a"]

    num_pages = 100000
    total_found_words = defaultdict(lambda: 0)
    for page_num in tqdm(range(1, num_pages + 1)):
        # Fetch a random Wikipedia page title
        random_title = get_random_wikipedia_title()
        if random_title:
            # Fetch the Wikipedia page content
            page_content = get_wikipedia_page(random_title)

            # Find sentences containing the predetermined words
            try:
                sentences, found_words, found_word_counts = find_instances_of_words(page_content, predetermined_words)
            except ValueError:
                continue
            total_found_words.update(found_word_counts)

            # Print the sentences for this page
            # print(f"Page {page_num}: {random_title}")
            # print(found_words, sentences)
            with open("scraped_sentences-3.txt", "a") as f:
                for sentence, found_word in zip(sentences, found_words):
                    f.write(found_word + " | " + sentence + "\n")
    print(total_found_words)


if __name__ == "__main__":
    main()
