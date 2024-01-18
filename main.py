import pandas as pd
import re
import nltk
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nltk.download('punkt')

class UserInput(BaseModel):
    user_input: str

class BookRecommendation:
    def __init__(self):
        # Veriyi okuma
        self.data = pd.read_csv("./data.csv", encoding="latin-1")

        # Kullanılmayacak verileri silmek
        self.data.drop(["isbn13", "isbn10", "published_year", "average_rating", "num_pages", "ratings_count"],
                       axis=1, inplace=True)

        # Boş değerleri silmek
        self.data.fillna("", inplace=True)

        # Veriyi birleştirme
        concat_data = (
            self.data['title'] + ' ' +
            self.data['authors'] + ' ' +
            self.data['categories'] + ' ' +
            self.data['description']
        )

        # DataFrameleri birleştirme
        self.merged_df = pd.concat([concat_data, self.data[["title", "thumbnail"]]], axis=1)

        # Sütun isimlerini güncelleme
        self.merged_df.columns = ["description", "title", "thumbnail"]

        # Tokenizer fonksiyonunu tanımlama
        self.porter_stemmer = PorterStemmer()

        # Tf-idf Vectorizer'ı oluşturma
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=self.stemming_tokenizer)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.merged_df["description"])

    def stemming_tokenizer(self, str_input):
        # Sadece harfleri ve tire karakterini koru, diğerleriyle değiştir
        words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
        # Her kelimenin kökünü al ve sayıları sıfırla
        stemmed_words = [self.porter_stemmer.stem(word) for word in words if word.isalpha()]
        return stemmed_words

    def get_book_recommendations(self, user_input):
        user_input_vectorized = self.tfidf_vectorizer.transform([user_input])

        # Cosine Similarity hesaplama
        cosine_similarities = cosine_similarity(user_input_vectorized, self.tfidf_matrix).flatten()

        # En çok benzeyen 10 kitabın indekslerini bulma
        top_10_similar_books_indices = cosine_similarities.argsort()[:-11:-1]

        # Tavsiye edilen 10 kitabın tam verilerini al
        recommended_books = []
        for index in top_10_similar_books_indices:
            recommended_book = self.data.iloc[index].to_dict()
            recommended_books.append(recommended_book)

        return recommended_books

app = FastAPI()
book_recommendation = BookRecommendation()

# CORS origin
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post('/api/book_recommendation')
async def get_book_recommendation(user_input: UserInput):
    recommendations = book_recommendation.get_book_recommendations(user_input.user_input)
    print(recommendations)
    return {'recommendations': recommendations}
