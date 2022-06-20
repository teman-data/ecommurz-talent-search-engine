from typing import List, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def load_model():
    """Load pretrained model from SentenceTransformer"""
    return SentenceTransformer('minilm_sbert')

def semantic_search(model, sentence, corpus_embeddings):
    """Perform semantic search on the corpus"""
    query_embeddings = model.encode(sentence,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True)

    hits = util.semantic_search(query_embeddings,
                                corpus_embeddings,
                                top_k=len(corpus_embeddings),
                                score_function=util.dot_score)

    return pd.DataFrame(hits[0])

def get_similarity_score(model, data, query, corpus_embeddings):
    """Get similarity score for each data point and sort by similarity score and day"""
    hits = semantic_search(model, [query], corpus_embeddings)
    result = pd.merge(data, hits, left_on='ID', right_on='corpus_id')
    result['Last Day'] = pd.to_datetime(result['Last Day'], format='%d/%m/%Y')
    result.sort_values(by=['score', 'Last Day'], ascending=[False, True], inplace=True)
    return result

@st.cache(allow_output_mutation=True)
def create_embedding(model: SentenceTransformer, data: pd.DataFrame, key: str) -> Tuple[list, list]:
    """Create vector embeddings from the dataset"""
    corpus_sentences = data[key].astype(str).tolist()
    corpus_embeddings = model.encode(sentences=corpus_sentences,
                                     show_progress_bar=True,
                                     convert_to_tensor=True,
                                     normalize_embeddings=True)
    return corpus_embeddings

def load_dataset(columns: List) -> pd.DataFrame:
    """Load real-time dataset from google sheets"""
    sheet_id = '1KeuPPVw9gueNmMrQXk1uGFlY9H1vvhErMLiX_ZVRv_Y'
    sheet_name = 'Form Response 3'.replace(' ', '%20')
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    data = pd.read_csv(url)
    data  = data.iloc[: , :7]
    data.columns = columns
    data.insert(0, 'ID', range(len(data)))
    return data

def show_aggrid_table(result: pd.DataFrame):
    """Show interactive table from similarity result"""
    gb = GridOptionsBuilder.from_dataframe(result)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(min_column_width=200)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
    gb.configure_column(field='LinkedIn Profile',
                        headerName='LinkedIn Profile',
                        cellRenderer=JsCode('''function(params) {return `<a href=${params.value} target="_blank">${params.value}</a>`}'''))

    grid_options = gb.build()

    grid_response = AgGrid(
        dataframe=result,
        gridOptions=grid_options,
        height=1100,
        fit_columns_on_grid_load=True,
        data_return_mode='AS_INPUT',
        update_mode='VALUE_CHANGED',
        theme='light',
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
    )

def main():
    """Main Function"""
    st.title('@ecommurz Talent Search Engine')
    st.write('This app lets you search and sort talent by job title or relevant job descriptions from ecommurz talent list in real-time.')

    columns = ['Timestamp', 'Full Name', 'Company', 'Previous Role',
               'Experience', 'Last Day', 'LinkedIn Profile']
    data = load_dataset(columns)
    model = load_model()
    corpus_embeddings = create_embedding(model, data, 'Previous Role')

    job_title = st.text_input('Insert the job title below:', '')
    submitted = st.button('Submit')

    if submitted:
        st.info(f'Showing results for {job_title}')
        result = get_similarity_score(model, data, job_title, corpus_embeddings)
        result = result[columns]
        show_aggrid_table(result)

if __name__ == '__main__':
    main()
