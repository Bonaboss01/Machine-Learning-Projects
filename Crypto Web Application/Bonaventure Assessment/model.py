
import pickle
import app
from sklearn.model_selection import train_test_split

from prophet import Prophet
fb = Prophet(interval_width=0.95, n_changepoints=7)
fb.fit(app.df_train)

# save the model
pickle_out = open('../model.pkl', 'wb')
pickle.dump(fb, pickle_out)
pickle_out.close()

