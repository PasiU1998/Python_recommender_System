from typing import Any, Union

from pandas import Series, DataFrame

import detect_keywords
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# TODO : loaded dummy data. Replace with actual data
dataset = detect_keywords.dataArr

dtm = detect_keywords.getDtmFromKeywords(dataset)

train = dtm[:2]
test: Union[Union[Series, DataFrame, None], Any] = dtm[2:]

classifier = MultinomialNB()
# TODO : replace 'string' with the binary class
classifier.fit(train, train['string'])

x_test = test
classifier.score(x_test, test['string'])

# Prediction test
predict_sentiment = classifier.predict(x_test)
test.loc['prediction', :] = predict_sentiment
print(test)
