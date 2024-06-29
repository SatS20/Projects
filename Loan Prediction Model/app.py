from fastapi import FastAPI
from pydantic import BaseModel #for defining the data types
import joblib
import numpy as np
import uvicorn 

class InputData(BaseModel):
    x1:float
    x2:float
    x3:float
    x4:float
    x5:float

#Load the model
scaler=joblib.load("Scaler.pkl")
model=joblib.load("model.pkl")
        
app=FastAPI()       

#define the endpoint 'predict' for API which is going to return prediction for our model
@app.post("/predict/") 
def predict(inputdata : InputData):
    x_values=np.array([[
        inputdata.x1,
        inputdata.x2,
        inputdata.x3,
        inputdata.x4,
        inputdata.x5
    ]])
    #inside the function we are going to scale our x values
    scaled_x_values=scaler.transform(x_values)
    prediction=model.predict(scaled_x_values)
    prediction=int(prediction[0])
    return {"prediction": prediction}


if __name__ =="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)



'''
The model.predict() function returns a numpy.ndarray (you could verify that using print(type(new_prediction))). You can't just return it in that format; hence, the Internal Server Error.

Option 1 is to simply retireve and return the first element of that numpy array:
return {'prediction': new_prediction[0]}

Option 2 is to convert the numpy array into a Python list using the .tolist() method.
return {'prediction': new_prediction.tolist()}
'''