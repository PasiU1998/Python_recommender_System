import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def getKeywords():
    keywords_file = open("keywords.txt", "r")
    keywords = []
    for keyword in keywords_file:
        keywords.append(keyword.split('\n')[0])
    return keywords

def getDTM(dataArr):
    dataObj = {}
    i = 0
    for data in dataArr:
        i += 1
        dataObj[''+str(i)] = [data]

    print(dataObj)

    df1 = pd.DataFrame(dataObj)

    # Initialize
    vectorizer = TfidfVectorizer()
    doc_vec = vectorizer.fit_transform(df1.iloc[0])

    # Create dataFrame
    df2 = pd.DataFrame(doc_vec.toarray().transpose(),
                       index=vectorizer.get_feature_names())

    # Change column headers
    df2.columns = df1.columns
    return df2

def getDtmFromKeywords(dataStringArr):
    dtm = []
    i = 0
    for dataString in dataStringArr:
        i += 1
        keywords = getKeywords()
        dtmObj = {'string': i}
        for keyword in keywords:
            dtmObj[''+str(keyword)] = str(dataString).lower().count(str(keyword).lower())
        dtm.append(dtmObj)
    return pd.DataFrame(dtm)

# Test DTM
dataArr = [
"Go is typed statically compiled language. It was created by Robert Griesemer, Ken Thompson, and Rob Pike in 2009. This language offers garbage collection, concurrency of CSP-style, memory safety, and structural typing.",
"Java is a language for programming that develops a software for several platforms. A compiled code or bytecode on Java application can run on most of the operating systems including Linux, Mac operating system, and Linux. Most of the syntax of Java is derived from the C++ and C languages.",
"Python supports multiple programming paradigms and comes up with a large standard library, paradigms included are object-oriented, imperative, functional and procedural."
]
# print(getDTM(dataArr))

