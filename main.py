from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import requests
from PIL import Image
import json
import torch


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Image_URL : str


# loading the model1 saved model
loaded_model1 = pickle.load(open('ic_model_1.sav','rb'))
#loading the model1 feature extractor
loaded_fe1 = pickle.load(open('ic_fe_1', 'rb'))
#loading the model1 tokenizer
loaded_tkn1 = pickle.load(open('ic_tkn_1', 'rb'))
# loading the model2 saved model
loaded_model2 = pickle.load(open('ic_model_2.sav','rb'))
# loading the model2 processor
loaded_prc2 = pickle.load(open('ic_prc_2','rb'))


@app.post('/caption_image')
def captioning_image(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    image_url = input_dictionary['Image_URL']
    
    max_length = 16
    num_beams = 4 
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    i_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                                ##MODEL1 PREDICTION##
    pixel_values = loaded_fe1(images=[i_image], return_tensors="pt").pixel_values
    output_ids = loaded_model1.generate(pixel_values, **gen_kwargs)
    
    preds = loaded_tkn1.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    pred1 = preds[0]
                                ##MODEL2 PREDICTION## 
    input_model2 = loaded_prc2(i_image, return_tensors="pt")

    out = loaded_model2.generate(**input_model2)
    pred2 = loaded_prc2.decode(out[0], skip_special_tokens=True)
    return [pred1,pred2]