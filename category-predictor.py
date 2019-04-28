from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 
        'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics', 
        'sci.med': 'Medicine'}

training_data = fetch_20newsgroups(subset='train',categories=category_map.keys(), shuffle=True, random_state=5)

count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)


input_string = raw_input("enter the string")
input_string1 = [input_string]

classifier = MultinomialNB().fit(train_tfidf, training_data.target)

input_tc = count_vectorizer.transform(input_string1)

input_tfidf = tfidf.transform(input_tc)
predictions = classifier.predict(input_tfidf)

for category in predictions:
    print(category_map[training_data.target_names[category]])

"""with open('KTcategory-predictor.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('KTtfidf.pkl','wb') as file1:
    pickle.dump(tfidf, file1)
    
with open('KTcv.pkl','wb') as file2:
    pickle.dump(train_tc,file2)
 """