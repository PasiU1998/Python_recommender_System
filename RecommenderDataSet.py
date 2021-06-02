import pandas as pd
import detect_keywords

keywords = detect_keywords.getKeywords()
occupations = ["Software Developer","Business Analyst","Electrician","Youtuber","Businessman","Mechanic","Musician",
               "Architect","Electrician","Student","Lecturer","Lecturer","Management analyst","Tailor"]

def getOccupationIndex(text):
    i = 0
    for occupation in occupations:
        if text == occupation:
            return i
        i += 1
    return 0

def getOccupationByIndex(i):
    return occupations[i]

def getGenderIndex(text):
    if text == "Male" or text == "male":
        return 0
    else:
        return 1

def getGenderByIndex(i):
    if i == 0:
        return "Male"
    else:
        return "Female"


def getKeywordCount(text, keyword):
    return 0
    keyword = str(keyword).lower()
    print(str(text).lower().count(keyword))
    return str(text).lower().count(keyword)


recommender_dataset = pd.read_csv('RecommenderDataset.csv')

for index, row in recommender_dataset.iterrows():
    datasetRowStr = str(row['Age'])
    datasetRowStr += "," + str(getGenderIndex(str(row['Gender'])))
    datasetRowStr += "," + str(getOccupationIndex(str(row['Occupation'])))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'product'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'brand'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'positive'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'negative'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'quality'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'price'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'software'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'application'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'app'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'language'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'library'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'location'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'program'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'business'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'school'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'food'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'event'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'design'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'plant'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'paint'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'car'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'travel'))
    datasetRowStr += "," + str(getKeywordCount(str(row['Visited poll names']) + str(row['Voted poll name']), 'system'))
    datasetRowStr += "," + str(0)
    print(datasetRowStr)

