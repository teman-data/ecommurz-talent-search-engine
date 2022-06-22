# @ecommurz Talent Search Engine
---
### 💡 How it Works
Before we dive into details, make sure to understand the basic concepts of semantic search which we explained in details on the [ICD10 suggestion project](https://github.com/teman-data/icd10-suggestion-engine). The main difference is that, in this project, we don't take top `k` entries from the corpus but we actually compute the similarity score for all of the corpus that we have. Besides that, we also did not save the embeddings because this is a live dataset. This project obviously has higher time complexity but is still feasible because of the small amount of data (< 1000 entries).

### 🤖 Model
For the model, we do not use paraphrasing model like the previous project but we opt for something small in size but with moderate accuracy. Based on the [available models](https://www.sbert.net/docs/pretrained_models.html) from SentenceTransformer, we can see that `all-mpnet-base-v2 model` provides the best quality, while `all-MiniLM-L6-v2` is 5 times faster and still offers good quality. Therefore, we choose `all-MiniLM-L6-v2` which is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for semantic search, job title, in this case. 

### 🤗 HuggingFace Spaces
[Hugging Face Spaces](https://huggingface.co/spaces/launch) offer a simple way to host ML demo apps directly on HF. This allows us to create your ML portfolio, showcase our projects at conferences or to stakeholders, and work collaboratively with other people in the ML ecosystem. Besides Spaces, there are also lternatives such as Heroku and Netlify but the reason we chose Spaces is mainly because of the following reasons:
- Support Streamlit SDK
- Free 16GB RAM and 8 CPU cores
- Control versioning out of the box and git-based workflows
- Accessible one link away
- Allows admins to set user roles to control access to repositories
