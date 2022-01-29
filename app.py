import streamlit as st
import pandas as pd
import re
import json
import torch

from models import Config, BERTModel, Tokenizer


@st.cache()
def get_movies_df():
    movies = pd.read_csv("artifacts/movie_ratings.csv")
    movies.index = movies.mid.astype('object')
    del movies["Unnamed: 0"]
    del movies["mid"]
    
    return movies


@st.cache()
def get_models():
    with open("artifacts/config.json") as f:
        config = Config(**json.load(f))

    with open("artifacts/smap.json") as f:
        smap = json.load(f)

    tk = Tokenizer(config, smap)
    bert = BERTModel.load_trained("artifacts/config.json", "artifacts/best_acc_model.pth")
    
    return tk, bert

movies = get_movies_df()
tokenizer, bert = get_models()


st.title("MovieLens1M Recommendation")
st.write("Movie List")
st.dataframe(movies.sort_values("rating", ascending=False))


txt = st.text_area('당신이 좋아한 영화의 ID를 입력해주세요(쉼표로 구분)', '''1193, 661, 914, 3408, 2355, 1197, 1287, 2804, 594, 919, 595, 938, 2398, 2918, 1035, 2791, 2687, 2018, 3105, 2797, 2321, 720, 1270, 527, 2340, 48, 1097, 1721, 1545, 745, 2294, 3186, 1566, 588, 1907, 783, 1836, 1022, 2762, 150, 1, 1961'''.strip())
# 1962, 2692, 260, 1028, 1029, 1207, 2028, 531, 3114, 608, 1246


sid_list = re.split(",\s", txt)

if len(sid_list) < 10:
    st.error("10개 이상의 영화를 골라주세요.")
else:
    st.write("당신이 좋아한 영화들")
    st.dataframe(movies.loc[[int(x) for x in sid_list], :])

    st.write("그런 당신을 위한 영화 TOP 20")
    inputs = tokenizer.encode(sid_list, insert_mask_token_last=True)
    values, indices = bert.predict_topk(torch.LongTensor([inputs]), k=20)
    sids = tokenizer.decode([str(x) for x in indices[0].detach().numpy()])
    result = pd.DataFrame({"score": values[0].detach().numpy()}, index=sids)
    result = movies.merge(result, left_index=True, right_index=True)
    st.write(result.sort_values("score", ascending=False))