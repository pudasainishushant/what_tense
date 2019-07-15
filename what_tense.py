from io import StringIO
import pandas as pd
import numpy as np
import spacy
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class TenseCategorizer():

	def __init__(self):
		self.nlp = spacy.load('en_core_web_sm')
		self.df_1,self.df_2 = self.prepare_data()
		self.X_train,self.X_test = np.array(self.preparation(self.df_1)),np.array(self.preparation(self.df_2))
		self.Y_train,self.Y_test = np.array(list(self.df_1["LABEL"])), np.array(list(self.df_2["LABEL"]))
		self.clf=RandomForestClassifier(n_estimators=100)
		self.fittingData()
		self.predicting()


	def fittingData(self):
		self.clf.fit(self.X_train,self.Y_train)

	def predicting(self):
		y_pred=self.clf.predict(self.X_test)
		conf_mat = confusion_matrix(self.Y_test,y_pred)
		#print(f"Confusion matrix obtained while using Random Forest Classifier{conf_mat}")
		return conf_mat


	def prepare_data(self):
		past_data = pd.read_csv("past_tense_data.csv",sep="\t",index_col=False)
		present_data = pd.read_csv("present_tense_data.csv",sep="\t",index_col=False)
		future_data = pd.read_csv("future_tense_data.csv",sep="\t",index_col=False)
		tense_data_list = [past_data,present_data,future_data]
		final_df = pd.concat(tense_data_list)
		final_df.loc[:,"SENTENCE"] = final_df.SENTENCE.apply(lambda x : str(x))
		for l in final_df["SENTENCE"]:
			if type(l) != str:
				print("bad")
		final_df=final_df.sample(frac=1).reset_index(drop=False)
		final_df_1 = final_df[:570]
		final_df_2 = final_df[:25]
		return final_df_1,final_df_2


	def preparation(self,df):
		texts = df["SENTENCE"].tolist()
		all_training_features = ['PRP', 'NN', 'IN', 'DT', 'VBN', 'VBG', 'VBD', 'RB', 'VB', 'JJ', 'NNS', 'NNP', 'VBP', 'MD', 'VBZ', 'TO', 'CD', 'WRB', 'RP', 'CC', 'PDT', 'JJS', 'WP', 'RBR', 'JJR', 'EX']
		all_training_features_binary = []
		for text in texts:
			doc = self.nlp(text)
			tokens = [t.text for t in doc]
			pos_tags = [t.tag_ for t in doc]
			sentence_feature = [int(all_training_features[i] in pos_tags) for i in range(len(all_training_features))]
			all_training_features_binary.append(sentence_feature)
		return all_training_features_binary



	def enterSentence(self,user_sent):
		#user_input = input('Enter text here!!')
		test_data = StringIO(f"""SENTENCE
		{user_sent}
		""")
		test_dataframe = pd.read_csv(test_data, sep=";")
		test = np.array(self.preparation(test_dataframe))
		value_required = self.clf.predict(test)
		#print(self.clf.predict(test))
		return value_required





#tense = TenseCategorizer()
#tense.enterSentence()
