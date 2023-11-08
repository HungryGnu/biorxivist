# BioRxivist

BioRxivist is a tool designed to help obtain full text articles for [BioRxiv](https://biorxiv.org) and facilitate the use of this text to integrate with LLM and build knowledge graph infrastructure using [Neo4J](https://neo4j.com/).



```python
import os
import sys
sys.version
```




    '3.10.13 (main, Nov  6 2023, 22:35:59) [GCC 9.4.0]'




```python
from dotenv import find_dotenv, load_dotenv
from os import environ
import openai
# import classes from BioRxivist
from biorxivist.webtools import BioRxivPaper, BioRxivDriver, SearchResult
from biorxivist.vectorstore import Neo4JDatabase
```


```python
# load_environment variables like our API key and Neo4J credentials
load_dotenv(find_dotenv())
```




    True



# 1. Using BioRxivist to find and load text from papers

The first thing we'll want to do is find a paper of interest.  Well use the object from this package. `BioRxivDriver` will give us access to BioRxiv's search utility. `SearchResuls` will help us manage the results of our search. `BioRxivPaper` will manage how we access text from the papers


```python
driver = BioRxivDriver()
```


```python
r = driver.search_biorxiv('TGF-Beta 1 Signaling in monocyte maturation')
```


```python
r.response.request.url
```




    'https://www.biorxiv.org/search/TGF-Beta+1+Signaling+in+monocyte+maturation%20numresults:75'




```python
type(r)
```




    biorxivist.webtools.SearchResult




```python
len(r.results)
```




    75




```python
# load more results
r.more()
```


```python
len(r.results)
```




    150




```python
# Results are a list of BioRxivPaper objects
r.results[0:5]
```




    [[Cross-species analysis identifies conserved transcriptional mechanisms of neutrophil maturation](https://biorxiv.org/content/10.1101/2022.11.28.518146v1),
     [Melanogenic Activity Facilitates Dendritic Cell Maturation via FMOD](https://biorxiv.org/content/10.1101/2022.05.14.491976v2),
     [Microglia integration into human midbrain organoids leads to increased neuronal maturation and functionality](https://biorxiv.org/content/10.1101/2022.01.21.477192v1),
     [Long-term culture of fetal monocyte precursors in vitro allowing the generation of bona fide alveolar macrophages in vivo](https://biorxiv.org/content/10.1101/2021.06.04.447115v2),
     [Pathologic α-Synuclein Species Activate LRRK2 in Pro-Inflammatory Monocyte and Macrophage Responses](https://biorxiv.org/content/10.1101/2020.05.04.077065v1)]



# BioRxivPaper Objects:
BioRxiv Paper objects link us to BioRxiv resources related to individual papers. Once instantiated these papers load a minimal amount of information into memory the URI of the papers homepage and the title.  Other features like the paper's abstract and full text are lazy-loaded properties. They are only accessed once we call them for the first time. After that they are available from memory so we don't have to hit the URL another time.

They are also feed directly into our LangChain pipeline.  In the next section we will make sure of their BioRxivPaper.langchain_html_doc attribute.

# Interact with individual papers:
The results are a collection of BioRxivPaper objects.  We can interact with them by indexing into the list or we can pull them out and interact with them here:


```python
paper3 = r.results[3]
```


```python
paper3.title
```




    'Long-term culture of fetal monocyte precursors in vitro allowing the generation of bona fide alveolar macrophages in vivo'




```python
paper3.abstract
```




    'Tissue-resident macrophage-based immune therapies have been proposed for various diseases. However, generation of sufficient numbers that possess tissue-specific functions remains a major handicap. Here, we show that fetal liver monocytes (FLiMo) cultured with GM-CSF (also known as CSF2) rapidly differentiate into a long-lived, homogeneous alveolar macrophage (AM)-like population in vitro. CSF2-cultured FLiMo remain the capacity to develop into bona fide AM upon transfer into Csf2ra-/- neonates and prevent development of alveolar proteinosis and efferocytosis of apoptotic cells for at least 1 year in vivo. Compared to transplantation of AM-like cells derived from bone marrow macrophages (BMM), CSF2-cFliMo more efficiently engraft empty AM niches in the lung and protect mice from respiratory viral infection. Harnessing the potential of this approach for gene therapy, we restored a disrupted Csf2ra gene in FLiMo and their capacity to develop into AM in vivo. Together, we provide a novel platform for generation of immature AM-like precursors amenable for genetic manipulation, which will be useful to study to dissect AM development and function and pulmonary transplantation therapy.'



# Accessing the paper's text

There are now a few ways to access the papers text:


```python
# through the paper.text property:
# print(paper3.text)
# by accessing the BioRxivPaper.__str__ attribute:
print(f'...{paper3[392:1000]}...')
```

    ... CSF2-cultured FLiMo remain the capacity to develop into bona fide AM upon transfer into Csf2ra-/- neonates and prevent development of alveolar proteinosis and efferocytosis of apoptotic cells for at least 1 year in vivo. Compared to transplantation of AM-like cells derived from bone marrow macrophages (BMM), CSF2-cFliMo more efficiently engraft empty AM niches in the lung and protect mice from respiratory viral infection. Harnessing the potential of this approach for gene therapy, we restored a disrupted Csf2ra gene in FLiMo and their capacity to develop into AM in vivo. Together, we provide a novel ...


# Build vector Embeddings:


```python
openai.api_key = environ['OPENAI_API_KEY']
```


```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# TODO: There is an HTML text splitter. maybe skip bs4 and cut to chase?
from langchain.vectorstores import Neo4jVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
```


```python
len(paper3.langchain_doc)
```

    Created a chunk of size 1197, which is longer than the specified 1000
    Created a chunk of size 1274, which is longer than the specified 1000
    Created a chunk of size 1647, which is longer than the specified 1000
    Created a chunk of size 1902, which is longer than the specified 1000
    Created a chunk of size 1529, which is longer than the specified 1000
    Created a chunk of size 1496, which is longer than the specified 1000
    Created a chunk of size 1123, which is longer than the specified 1000
    Created a chunk of size 1078, which is longer than the specified 1000
    Created a chunk of size 1122, which is longer than the specified 1000
    Created a chunk of size 1342, which is longer than the specified 1000
    Created a chunk of size 1292, which is longer than the specified 1000
    Created a chunk of size 1133, which is longer than the specified 1000
    Created a chunk of size 1627, which is longer than the specified 1000
    Created a chunk of size 1483, which is longer than the specified 1000
    Created a chunk of size 1387, which is longer than the specified 1000
    Created a chunk of size 1177, which is longer than the specified 1000
    Created a chunk of size 1695, which is longer than the specified 1000
    Created a chunk of size 1495, which is longer than the specified 1000
    Created a chunk of size 1233, which is longer than the specified 1000
    Created a chunk of size 1192, which is longer than the specified 1000
    Created a chunk of size 1187, which is longer than the specified 1000
    Created a chunk of size 1174, which is longer than the specified 1000





    43




```python
embeddings = OpenAIEmbeddings()
```


```python
# TODO lets make an object in BioRxivist that does this
vec = Neo4jVector.from_documents(
    paper3.langchain_doc, OpenAIEmbeddings(),
    url=f'bolt://{environ["NEO4J_HOST"]}:{environ["NEO4J_BOLT_PORT"]}',
    username=environ['NEO4J_USERNAME'],
    password=''
)
```


```python
type(vec)
```




    langchain.vectorstores.neo4j_vector.Neo4jVector




```python
docs_with_score = vec.similarity_search_with_score('What is the role of CSF2?', k=5)
```


```python
docs_with_score[0]
```




    (Document(page_content='In addition to the homeostatic function, AM play an essential role in protecting influenza virus-infected mice from morbidity by maintaining lung integrity through the removal of dead cells and excess surfactant (Schneider et al, 2014). To assess the functional capacity of CSF2-cFLiMo-derived AM during pulmonary virus infection, we reconstituted Csf2ra-/- neonates with CSF2-cFLiMo and infected adults 10 weeks later with influenza virus PR8 (Fig. 5A). Without transfer, Csf2ra-/- mice succumbed to infection due to lung failure (Fig. 5B-E), as reported previously (Schneider et al, 2017). Notably, the presence of CSF2-cFLiMo-derived-AM protected Csf2ra-/- mice from severe morbidity (Fig. 5B, C) and completely restored viability (Fig. 5D) and O2 saturation (Fig. 5E) compared to infected WT mice.', metadata={'title': 'Long-term culture of fetal monocyte precursors in vitro allowing the generation of bona fide alveolar macrophages in vivo', 'source': 'https://biorxiv.org/content/10.1101/2021.06.04.447115v2.full-text'}),
     0.9213951230049133)




```python
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
```

    --------------------------------------------------------------------------------
    Score:  0.9213951230049133
    In addition to the homeostatic function, AM play an essential role in protecting influenza virus-infected mice from morbidity by maintaining lung integrity through the removal of dead cells and excess surfactant (Schneider et al, 2014). To assess the functional capacity of CSF2-cFLiMo-derived AM during pulmonary virus infection, we reconstituted Csf2ra-/- neonates with CSF2-cFLiMo and infected adults 10 weeks later with influenza virus PR8 (Fig. 5A). Without transfer, Csf2ra-/- mice succumbed to infection due to lung failure (Fig. 5B-E), as reported previously (Schneider et al, 2017). Notably, the presence of CSF2-cFLiMo-derived-AM protected Csf2ra-/- mice from severe morbidity (Fig. 5B, C) and completely restored viability (Fig. 5D) and O2 saturation (Fig. 5E) compared to infected WT mice.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.92054283618927
    CSF2-cFLiMo generated from wild-type or gene-deficient mice could be used as a high-throughput screening system to study AM development in vitro and in vivo. Our model is suitable to study the relationship between AM and lung tissue, as well as the roles of specific genes or factors in AM development and function. Furthermore, CSF2-cFLiMo can overcome the limitation in macrophage precursor numbers and be used as a therapeutic approach for PAP disease or in other macrophage-based cell therapies including lung emphysema, lung fibrosis, lung infectious disease and lung cancer (Byrne et al, 2016; Lee et al, 2016; Wilson et al, 2010). Finally, genetically modified and transferred CSF2-cFLiMo might facilitate the controlled expression of specific therapeutic proteins in the lung for disease treatment, and therefore, could represent an attractive alternative to non-specific gene delivery by viral vectors.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.9168769717216492
    Overall, our studies demonstrate that CSF2-cFLiMo-AM were functionally equivalent to naturally differentiated AM. To determine the number of donor cells required to fully reconstitute the AM compartment of Csf2ra-/- mice, we titrated the number of transferred CSF2-cFLiMo (Fig. 4A). Transfer of a minimum of 5×104 CSF2-cFLiMo to neonatal Csf2ra-/- mice resulted in AM numbers in adult recipients that were comparable to unmanipulated WT mice (around 5×105) (Fig. 4B) and protected mice from PAP (Fig. 4C). We have previously established that around 10% of primary fetal liver monocytes supplied intranasally reach the lung (Li et al, 2020). Thus, CSF2-cFLiMo have expanded around 100-fold 6 weeks after transfer to Csf2ra-/- neonates. Notably, extended time of CSF2-cFLiMo in vitro culture (i.e. 4 months) prior transfer into recipient mice did not negatively affect their differentiation and functional capacity (Fig. 4B, C). Another critical function of tissue-resident macrophages including AM is the removal of apoptotic cells (efferocytosis) (Morioka et al, 2019). We compared efferocytosis between CSF2-cFLiMo-AM in Csf2ra-/- mice and AM in WT mice by intratracheal (i.t.) installation of labelled apoptotic thymocytes. CSF2-cFLiMo-AM and AM were equally potent at phagocytosing apoptotic cells from the bronchoalveolar space (Fig. 4D).
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.9152920246124268
    Next, we assessed whether CSF2-cFLiMo show therapeutic activity upon transfer into adult Csf2ra-/- mice, which had already developed PAP. Adult Csf2ra-/- mice were transferred i.t. with 0.5, 1 or 2 million CSF2-cFLiMo (Fig. 4E-G). Ten weeks after transfer, donor-derived AM were detectable in the BAL and lung of Csf2ra-/- only in recipients transferred with 2 million cells (Fig. 4F). The protein levels in the BAL from mice transferred with 2×106 cells were significantly lower when compared to naïve Csf2ra-/- mice, suggesting that transferred cells were able to reduce proteinosis, although not to the level of WT mice (Fig. 4G). However, CSF2-cFLiMo-derived AM exhibited higher expression of F4/80 and CD11b, and lower expression of Siglec-F and CD64 when compared to WT AM (Fig. E5A, B), indicating that the AM phenotype was not fully recapitulated but intermediate between AM-derived from CSF2-cFLiMo transferred to neonates and AM-derived from CSF2-cFLiMo transplanted to adult mice. These results show that CSF2-cFLiMo can reproduce AM phenotype and function most adequately only when transferred to neonatal Csf2ra-/- mice.
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
    Score:  0.9150279760360718
    To assess whether CSF2-cFLiMo can develop to bona fide AM and perform AM function in vivo, we transferred congenically marked CSF2-cFLiMo intranasally (i.n.) to newborn Csf2ra-/- mice (Fig. 1A), which lack AM (Li et al, 2020; Schneider et al, 2017). Analysis of the BAL and lung of Csf2ra-/- recipients showed efficient engraftment of donor-derived cells that resemble mature CD11chiSiglec-Fhi AM (Fig. 1C and Fig. E3A). The numbers of CSF2-cFLiMo-derived AM rapidly increased within the first 6 weeks after transfer, before reaching a relatively stable population size (Fig. 1D), similar to the kinetics during normal postnatal AM differentiation (Guilliams et al, 2013; Schneider et al, 2014). While CSF2-cFLiMo were CD11bhiSiglec-Flo before transfer, they down-regulated CD11b and up-regulated CD11c and Siglec-F surface expression upon transfer and expansion in vivo, indicating that they completed their differentiation to become cells with a phenotype that is indistinguishable from AM of age-matched WT mice (Fig. 1E). Notably, CSF2-cFLiMo-derived AM were maintained in the lung for at least 1 year after transfer (Fig. 1C, D). Moreover, measurement of protein concentration in the BAL at different time points after transfer showed that CSF2-cFLiMo-AM reconstituted Csf2ra-/- mice were completely protected from PAP up to one year (Fig. 1F and Fig. E3B). These results demonstrate that CSF2-cFLiMo develop into mature AM, which appear functionally equivalent to in situ differentiated AM.
    --------------------------------------------------------------------------------


# Connect to an existing vector Store


```python
db = Neo4JDatabase.from_environment()
```


```python
db.url
```




    'neo4j://localhost:7687'




```python
labels = db.fetch_labels_and_properties()
```


```python
labels
```




    [{"labels": ["Chunk"], "propkeys": ["id", "text", "title", "source", "embedding"]}]




```python
labels.data
```




    [{'labels': ['Chunk'],
      'propkeys': ['id', 'text', 'title', 'source', 'embedding']}]




```python
labels.metadata.query_type
```




    'r'




```python
print(labels)
# This tells us the label and property keys we want when we make our vector.
```

    [{"labels": ["Chunk"], "propkeys": ["id", "text", "title", "source", "embedding"]}]



```python
with db as session:
    r = session.execute_read(db.transaction, "SHOW CONSTRAINTS")
r
# this tells us that "id" is an index
```




    [{"id": 5, "name": "constraint_1dc138a", "type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": ["Chunk"], "properties": ["id"], "ownedIndex": "constraint_1dc138a", "propertyType": null}]




```python
vec2 = db.make_vectorstore(
    embedding=OpenAIEmbeddings(),
    index_name="id",
    node_label="Chunk",
    text_node_properties=["id", "text", "title", "source", "embedding"],
    embedding_node_property="embedding"
)
```


```python
type(vec2)
```




    langchain.vectorstores.neo4j_vector.Neo4jVector




```python

```
